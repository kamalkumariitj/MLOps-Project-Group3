import argparse
from datetime import datetime
import json
import pickle
from pathlib import Path

from config import is_small_run, load_config
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from utils import TextClassificationDataset, compute_metrics, set_seed


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Evaluate trained classifier and save metrics report.")
    parser.add_argument("--run-mode", choices=["SMALL_RUN", "FULL_RUN"], default=cfg.run_mode)
    parser.add_argument("--data-path", default=cfg.data_pickle_path)
    parser.add_argument("--model-dir", default=cfg.output_dir)
    parser.add_argument("--max-length", type=int, default=cfg.max_length)
    parser.add_argument("--eval-batch-size", type=int, default=cfg.eval_batch_size)
    parser.add_argument("--report-path", default=cfg.eval_report_path)
    parser.add_argument("--seed", type=int, default=cfg.seed)

    parser.add_argument("--enable-wandb", dest="enable_wandb", action="store_true", default=cfg.enable_wandb)
    parser.add_argument("--disable-wandb", dest="enable_wandb", action="store_false")
    parser.add_argument("--wandb-project", default=cfg.wandb_project)
    parser.add_argument("--wandb-entity", default=cfg.wandb_entity)
    parser.add_argument("--wandb-api-key", default=cfg.wandb_api_key)
    parser.add_argument("--wandb-run-name", default=cfg.wandb_run_name)
    parser.add_argument(
        "--log-eval-report-to-wandb",
        dest="wandb_log_eval_report",
        action="store_true",
        default=cfg.wandb_log_eval_report,
    )
    parser.add_argument("--no-log-eval-report-to-wandb", dest="wandb_log_eval_report", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    small_run = is_small_run(args.run_mode)

    with open(args.data_path, "rb") as f:
        data = pickle.load(f)

    test_texts = data["test_texts"]
    test_labels_encoded = data["test_labels_encoded"]
    id2label = {int(k): str(v) for k, v in data["id2label"].items()}

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=args.max_length)
    test_dataset = TextClassificationDataset(test_encodings, test_labels_encoded)

    eval_args = TrainingArguments(
        output_dir=str(model_dir / "eval_tmp"),
        per_device_eval_batch_size=args.eval_batch_size,
        report_to="none",
        do_train=False,
        do_eval=True,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    eval_results = trainer.evaluate()
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)

    report = classification_report(
        test_labels_encoded,
        preds,
        target_names=[id2label[i] for i in range(len(id2label))],
        output_dict=True,
    )
    report["eval_loss"] = eval_results.get("eval_loss")
    report["eval_accuracy"] = eval_results.get("eval_accuracy")
    report["eval_f1"] = eval_results.get("eval_f1")

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Evaluation complete. Report saved to: {report_path}")
    print(f"accuracy={report['eval_accuracy']:.4f} f1={report['eval_f1']:.4f} loss={report['eval_loss']:.4f}")

    if args.enable_wandb and not small_run:
        import wandb

        if args.wandb_api_key:
            wandb.login(key=args.wandb_api_key, relogin=True)

        run_name = args.wandb_run_name or f"eval-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            job_type="evaluation",
            config={
                "model_dir": str(model_dir),
                "data_path": args.data_path,
                "report_path": str(report_path),
                "run_mode": args.run_mode,
            },
        )
        wandb.log(
            {
                "eval/loss": report["eval_loss"],
                "eval/accuracy": report["eval_accuracy"],
                "eval/f1": report["eval_f1"],
            }
        )
        if args.wandb_log_eval_report:
            artifact = wandb.Artifact("eval-report", type="evaluation")
            artifact.add_file(str(report_path))
            wandb.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    main()
