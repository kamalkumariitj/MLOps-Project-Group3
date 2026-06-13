import argparse
import json
import os
import warnings
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


def parse_input_texts(raw_input_text: str) -> list[str]:
    normalized = raw_input_text.replace("\\n", "\n")
    return [line.strip() for line in normalized.splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    cfg = load_config()
    if not args.input_text or not args.input_text.strip():
        raise ValueError(
            "INPUT_TEXT is required. "
            "Pass --input-text or set INPUT_TEXT environment variable."
        )

    device = resolve_device(args.device)
    model_name_in_use = cfg.inference_model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_in_use, token=args.hf_token)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_in_use, token=args.hf_token).to(device)
    except Exception as exc:
        if cfg.inference_fallback_model_name and cfg.inference_fallback_model_name != model_name_in_use:
            warnings.warn(
                f"Failed to load HF_MODEL_NAME='{model_name_in_use}'. Falling back to "
                f"'{cfg.inference_fallback_model_name}'. Error: {exc}"
            )
            model_name_in_use = cfg.inference_fallback_model_name
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_in_use, token=args.hf_token
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name_in_use, token=args.hf_token
            ).to(device)
        else:
            raise
    model.eval()

    input_texts = parse_input_texts(args.input_text)
    if not input_texts:
        raise ValueError(
            "INPUT_TEXT is required. "
            "Provide one or more non-empty lines."
        )

    encoded = tokenizer(
        input_texts,
        truncation=True,
        padding=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
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

    if (not id2label or labels_are_placeholders(id2label)) and probs.shape[-1] == len(ANLI_LABELS):
        id2label = {i: label for i, label in enumerate(ANLI_LABELS)}

    predictions = []
    for row_index, text in enumerate(input_texts):
        row_probs = probs[row_index]
        pred_idx = int(torch.argmax(row_probs).item())
        pred_label = id2label.get(pred_idx, f"LABEL_{pred_idx}")
        scores = {
            id2label.get(i, f"LABEL_{i}"): float(row_probs[i].item())
            for i in range(row_probs.shape[0])
        }
        predictions.append(
            {
                "input_text": text,
                "predicted_label": pred_label,
                "predicted_index": pred_idx,
                "scores": scores,
            }
        )

    if len(predictions) == 1:
        result = {
            "model_name": model_name_in_use,
            "input_text": predictions[0]["input_text"],
            "predicted_label": predictions[0]["predicted_label"],
            "predicted_index": predictions[0]["predicted_index"],
            "scores": predictions[0]["scores"],
            "device": str(device),
            "total_inputs": 1,
        }
    else:
        result = {
            "model_name": model_name_in_use,
            "device": str(device),
            "total_inputs": len(predictions),
            "results": predictions,
        }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
