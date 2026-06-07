import argparse
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

from datasets import load_dataset

from config import is_small_run, load_config
from utils import clean_split, create_label_maps, encode_labels, save_json, set_seed


def build_text(example: dict) -> str:
    return f"premise: {example['premise']} hypothesis: {example['hypothesis']}"


def collect_split(ds, split_name: str, max_rows: int, seed: int, label_id_to_name: dict) -> Tuple[List[str], List[str]]:
    split = ds[split_name].shuffle(seed=seed)
    split = split.select(range(min(max_rows, len(split))))
    texts = [build_text(ex) for ex in split]
    labels = [label_id_to_name[int(ex["label"])] for ex in split]
    return texts, labels


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Load, clean, sample, split and encode ANLI data.")
    parser.add_argument("--run-mode", choices=["SMALL_RUN", "FULL_RUN"], default=cfg.run_mode)
    parser.add_argument("--dataset-name", default=cfg.dataset_name)
    parser.add_argument("--train-splits", nargs="+", default=cfg.train_splits)
    parser.add_argument("--test-splits", nargs="+", default=cfg.test_splits)
    parser.add_argument("--train-max-rows", type=int, default=cfg.train_max_rows)
    parser.add_argument("--test-max-rows", type=int, default=cfg.test_max_rows)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--small-run", action="store_true", help="Use tiny slices for fast smoke checks.")
    parser.add_argument("--output-pickle", default=cfg.data_pickle_path)
    parser.add_argument("--label-map-path", default=cfg.label_map_path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.small_run or is_small_run(args.run_mode):
        args.train_splits = ["train_r1"]
        args.test_splits = ["test_r1"]
        args.train_max_rows = min(args.train_max_rows, 300)
        args.test_max_rows = min(args.test_max_rows, 100)

    print(f"Loading dataset: {args.dataset_name}")
    ds = load_dataset(args.dataset_name)

    label_names: Sequence[str] = ds["train_r1"].features["label"].names
    label_id_to_name = {i: name for i, name in enumerate(label_names)}
    label2id, id2label = create_label_maps(label_names)
    valid_labels = set(label2id.keys())

    all_train_texts, all_train_labels = [], []
    for split_name in args.train_splits:
        texts, labels = collect_split(ds, split_name, args.train_max_rows, args.seed, label_id_to_name)
        all_train_texts.extend(texts)
        all_train_labels.extend(labels)

    all_test_texts, all_test_labels = [], []
    for split_name in args.test_splits:
        texts, labels = collect_split(ds, split_name, args.test_max_rows, args.seed, label_id_to_name)
        all_test_texts.extend(texts)
        all_test_labels.extend(labels)

    print(f"Raw train={len(all_train_texts)} | raw test={len(all_test_texts)}")
    print(f"Raw train labels: {Counter([str(x).strip().lower() for x in all_train_labels])}")
    print(f"Raw test labels: {Counter([str(x).strip().lower() for x in all_test_labels])}")

    train_texts, train_labels, train_stats = clean_split(all_train_texts, all_train_labels, valid_labels=valid_labels)
    test_texts, test_labels, test_stats = clean_split(all_test_texts, all_test_labels, valid_labels=valid_labels)

    train_labels_encoded = encode_labels(train_labels, label2id)
    test_labels_encoded = encode_labels(test_labels, label2id)

    print(
        "Clean train:",
        train_stats,
        "| label distribution:",
        Counter(train_labels),
    )
    print(
        "Clean test:",
        test_stats,
        "| label distribution:",
        Counter(test_labels),
    )

    payload = {
        "train_texts": train_texts,
        "train_labels": train_labels,
        "train_labels_encoded": train_labels_encoded,
        "test_texts": test_texts,
        "test_labels": test_labels,
        "test_labels_encoded": test_labels_encoded,
        "label2id": label2id,
        "id2label": id2label,
    }

    output_pickle = Path(args.output_pickle)
    output_pickle.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pickle, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved processed dataset to: {output_pickle}")

    save_json({str(k): v for k, v in id2label.items()}, args.label_map_path)
    print(f"Saved id2label mapping to: {args.label_map_path}")


if __name__ == "__main__":
    main()
