import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class CleaningStats:
    total_in: int
    dropped_missing: int
    dropped_invalid_label: int
    dropped_duplicates: int
    total_out: int


class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: Dict[str, Sequence[int]], labels: Sequence[int]):
        self.encodings = encodings
        self.labels = list(labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_label_maps(label_names: Iterable[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    names = [str(x).strip().lower() for x in label_names]
    label2id = {name: idx for idx, name in enumerate(names)}
    id2label = {idx: name for name, idx in label2id.items()}
    return label2id, id2label


def normalize_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = str(text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    text = text.lower().strip()
    text = re.sub(r"[^\w\s:]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_split(
    texts: Sequence[str],
    labels: Sequence[str],
    valid_labels: Optional[set] = None,
) -> Tuple[List[str], List[str], CleaningStats]:
    cleaned_pairs: List[Tuple[str, str]] = []
    dropped_missing = 0
    dropped_invalid_label = 0

    for text, label in zip(texts, labels):
        text_norm = normalize_text(text)
        label_norm = str(label).strip().lower() if label is not None else None

        if not text_norm or label_norm is None:
            dropped_missing += 1
            continue

        if valid_labels is not None and label_norm not in valid_labels:
            dropped_invalid_label += 1
            continue

        cleaned_pairs.append((text_norm, label_norm))

    before_dedup = len(cleaned_pairs)
    cleaned_pairs = list(dict.fromkeys(cleaned_pairs))
    dropped_duplicates = before_dedup - len(cleaned_pairs)

    clean_texts = [x[0] for x in cleaned_pairs]
    clean_labels = [x[1] for x in cleaned_pairs]

    stats = CleaningStats(
        total_in=len(texts),
        dropped_missing=dropped_missing,
        dropped_invalid_label=dropped_invalid_label,
        dropped_duplicates=dropped_duplicates,
        total_out=len(clean_texts),
    )
    return clean_texts, clean_labels, stats


def encode_labels(labels: Sequence[str], label2id: Dict[str, int]) -> List[int]:
    return [label2id[str(label).strip().lower()] for label in labels]


def compute_metrics(pred) -> Dict[str, float]:
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def save_json(data: Dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
