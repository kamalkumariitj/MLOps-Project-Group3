import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import load_config

ANLI_LABELS = ["entailment", "neutral", "contradiction"]


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Run single-text inference with a Hugging Face classification model.")
    parser.add_argument("--input-text", default=os.getenv("INPUT_TEXT"))
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--label-map-path", default=cfg.label_map_path)
    parser.add_argument("--max-length", type=int, default=cfg.max_length)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_label_map(path: str | None) -> dict[int, str]:
    if not path:
        return {}
    map_path = Path(path)
    if not map_path.exists():
        return {}
    with open(map_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {int(k): str(v) for k, v in payload.items()}


def labels_are_placeholders(id2label: dict[int, str]) -> bool:
    return bool(id2label) and all(str(v).startswith("LABEL_") for v in id2label.values())


def main() -> None:
    args = parse_args()
    cfg = load_config()
    if not args.input_text or not args.input_text.strip():
        raise ValueError("INPUT_TEXT is required. Pass --input-text or set INPUT_TEXT environment variable.")

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.inference_model_name, token=args.hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.inference_model_name, token=args.hf_token).to(device)
    model.eval()

    encoded = tokenizer(
        [args.input_text.strip()],
        truncation=True,
        padding=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits[0]
        probs = torch.softmax(logits, dim=-1)

    id2label = {int(k): str(v) for k, v in (getattr(model.config, "id2label", {}) or {}).items()}
    file_id2label = load_label_map(args.label_map_path)
    if file_id2label and (not id2label or labels_are_placeholders(id2label)):
        id2label = file_id2label

    if not id2label or labels_are_placeholders(id2label):
        label2id = {str(k): int(v) for k, v in (getattr(model.config, "label2id", {}) or {}).items()}
        label2id_has_real_labels = label2id and any(not name.startswith("LABEL_") for name in label2id.keys())
        if label2id_has_real_labels:
            id2label = {idx: name for name, idx in label2id.items()}

    if (not id2label or labels_are_placeholders(id2label)) and probs.shape[0] == len(ANLI_LABELS):
        id2label = {i: label for i, label in enumerate(ANLI_LABELS)}

    pred_idx = int(torch.argmax(probs).item())
    pred_label = id2label.get(pred_idx, f"LABEL_{pred_idx}")
    scores = {id2label.get(i, f"LABEL_{i}"): float(probs[i].item()) for i in range(probs.shape[0])}

    result = {
        "model_name": cfg.inference_model_name,
        "input_text": args.input_text,
        "predicted_label": pred_label,
        "predicted_index": pred_idx,
        "scores": scores,
        "device": str(device),
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
