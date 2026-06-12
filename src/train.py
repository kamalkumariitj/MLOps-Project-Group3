import argparse
from datetime import datetime
import inspect
import json
import pickle
from pathlib import Path

from config import is_small_run, load_config
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from utils import TextClassificationDataset, compute_metrics, set_seed


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Train a Hugging Face sequence-classification model.")
    parser.add_argument("--run-mode", choices=["SMALL_RUN", "FULL_RUN"], default=cfg.run_mode)
    parser.add_argument(
        "--experiment-version",
        choices=sorted(cfg.experiment_configs.keys()),
        default=cfg.experiment_version,
    )
    parser.add_argument("--data-path", default=cfg.data_pickle_path)
    parser.add_argument("--model-name", default=cfg.model_name)
    parser.add_argument("--max-length", type=int, default=cfg.max_length)
    parser.add_argument("--output-dir", default=cfg.output_dir)
    parser.add_argument("--logging-dir", default=cfg.logging_dir)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--report-to", default="wandb" if cfg.enable_wandb else "none", choices=["none", "wandb"])
    parser.add_argument("--run-name", default=cfg.wandb_run_name)
    parser.add_argument("--small-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps. Use -1 for disabled.")

    parser.add_argument("--enable-wandb", dest="enable_wandb", action="store_true", default=cfg.enable_wandb)
    parser.add_argument("--disable-wandb", dest="enable_wandb", action="store_false")
    parser.add_argument("--wandb-project", default=cfg.wandb_project)
    parser.add_argument("--wandb-entity", default=cfg.wandb_entity)
    parser.add_argument("--wandb-api-key", default=cfg.wandb_api_key)

    parser.add_argument("--push-to-hub", dest="push_to_hub", action="store_true", default=cfg.push_to_hub)
    parser.add_argument("--no-push-to-hub", dest="push_to_hub", action="store_false")
    parser.add_argument("--hf-repo-id", default=cfg.hf_repo_id)
    parser.add_argument("--hf-token", default=cfg.hf_token)
    parser.add_argument("--push-tokenizer", dest="push_tokenizer", action="store_true", default=cfg.push_tokenizer)
    parser.add_argument("--no-push-tokenizer", dest="push_tokenizer", action="store_false")
    parser.add_argument(
        "--update-wandb-summary",
        dest="update_wandb_summary",
        action="store_true",
        default=cfg.update_wandb_summary,
    )
    parser.add_argument("--no-update-wandb-summary", dest="update_wandb_summary", action="store_false")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    experiment_cfg = cfg.experiment_configs.get(args.experiment_version, cfg.experiment_configs[cfg.experiment_version])
    args.epochs = experiment_cfg.epochs if args.epochs is None else args.epochs
    args.batch_size = experiment_cfg.batch_size if args.batch_size is None else args.batch_size
    args.eval_batch_size = experiment_cfg.eval_batch_size if args.eval_batch_size is None else args.eval_batch_size
    args.learning_rate = experiment_cfg.learning_rate if args.learning_rate is None else args.learning_rate
    args.warmup_steps = experiment_cfg.warmup_steps if args.warmup_steps is None else args.warmup_steps
    args.weight_decay = experiment_cfg.weight_decay if args.weight_decay is None else args.weight_decay
    args.max_steps = experiment_cfg.max_steps if args.max_steps is None else args.max_steps

    set_seed(args.seed)
    small_run = args.small_run or is_small_run(args.run_mode)

    with open(args.data_path, "rb") as f:
        data = pickle.load(f)

    train_texts = data["train_texts"]
    train_labels = data["train_labels_encoded"]
    test_texts = data["test_texts"]
    test_labels = data["test_labels_encoded"]
    label2id = {str(k): int(v) for k, v in data["label2id"].items()}
    id2label = {int(k): str(v) for k, v in data["id2label"].items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(id2label),
        label2id=label2id,
        id2label=id2label,
    )

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=args.max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=args.max_length)

    train_dataset = TextClassificationDataset(train_encodings, train_labels)
    test_dataset = TextClassificationDataset(test_encodings, test_labels)

    max_steps = args.max_steps
    eval_strategy = "epoch"
    save_strategy = "epoch"
    load_best_model_at_end = True

    if small_run:
        args.epochs = 1
        args.batch_size = 8
        args.learning_rate = 2e-5
        args.warmup_steps = 0
        args.report_to = "none"
        eval_strategy = "steps"
        save_strategy = "no"
        load_best_model_at_end = False
        if max_steps < 0:
            max_steps = 80

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.model_name}-{args.run_mode.lower()}-{timestamp}"
    wandb_run = None

    if args.enable_wandb and not small_run:
        import wandb

        if args.wandb_api_key:
            wandb.login(key=args.wandb_api_key, relogin=True)
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "run_mode": args.run_mode,
                "experiment_version": args.experiment_version,
                "model_name": args.model_name,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "learning_rate": args.learning_rate,
                "warmup_steps": args.warmup_steps,
                "weight_decay": args.weight_decay,
                "max_length": args.max_length,
                "max_steps": max_steps,
                "output_dir": args.output_dir,
            },
        )
        args.report_to = "wandb"
    else:
        args.report_to = "none"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_kwargs = {
        "output_dir": str(output_dir),
        "logging_dir": args.logging_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "logging_steps": 10 if small_run else 50,
        "save_strategy": save_strategy,
        "load_best_model_at_end": load_best_model_at_end,
        "report_to": args.report_to,
        "run_name": run_name if args.report_to == "wandb" else None,
        "seed": args.seed,
        "max_steps": max_steps,
    }
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        training_kwargs["eval_strategy"] = eval_strategy
    else:
        training_kwargs["evaluation_strategy"] = eval_strategy

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    train_result = trainer.train()
    eval_results = trainer.evaluate(eval_dataset=test_dataset)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved model and tokenizer to: {output_dir}")

    metrics = train_result.metrics
    metrics.update({f"final_{k}": v for k, v in eval_results.items()})
    metrics["train_samples"] = len(train_dataset)
    metrics["eval_samples"] = len(test_dataset)
    with open(output_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved training metrics to: {output_dir / 'train_metrics.json'}")

    if wandb_run is not None:
        import wandb

        wandb.log(
            {
                "final/eval_loss": eval_results.get("eval_loss"),
                "final/eval_accuracy": eval_results.get("eval_accuracy"),
                "final/eval_f1": eval_results.get("eval_f1"),
            }
        )

    if args.push_to_hub:
        if not args.hf_repo_id:
            raise ValueError(
                "HF repository id is required when push_to_hub is enabled. "
                "Set HF_REPO_ID in config/.env or pass --hf-repo-id."
            )
        from huggingface_hub import login

        if args.hf_token:
            login(token=args.hf_token)

        trainer.model.push_to_hub(args.hf_repo_id)
        if args.push_tokenizer:
            tokenizer.push_to_hub(args.hf_repo_id)
        hf_url = f"https://huggingface.co/{args.hf_repo_id}"
        print(f"Pushed model artifacts to: {hf_url}")

        if wandb_run is not None and args.update_wandb_summary:
            wandb_run.summary["huggingface_model"] = hf_url

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
