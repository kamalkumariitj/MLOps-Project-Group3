import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else int(value.strip())


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else float(value.strip())


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_list(name: str, default: List[str]) -> List[str]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return [x.strip() for x in value.split(",") if x.strip()]


@dataclass
class AppConfig:
    run_mode: str
    seed: int

    dataset_name: str
    train_splits: List[str]
    test_splits: List[str]
    train_max_rows: int
    test_max_rows: int
    data_pickle_path: str
    label_map_path: str

    model_name: str
    inference_model_name: str
    inference_fallback_model_name: str
    experiment_version: str
    experiment_configs: Dict[str, "ExperimentConfig"]
    max_length: int
    output_dir: str
    logging_dir: str
    eval_report_path: str
    epochs: int
    batch_size: int
    eval_batch_size: int
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    max_steps: int

    enable_wandb: bool
    wandb_project: str
    wandb_entity: Optional[str]
    wandb_run_name: Optional[str]
    wandb_log_eval_report: bool
    wandb_api_key: Optional[str]

    push_to_hub: bool
    hf_repo_id: Optional[str]
    hf_token: Optional[str]
    push_tokenizer: bool
    update_wandb_summary: bool


@dataclass
class ExperimentConfig:
    epochs: int
    batch_size: int
    eval_batch_size: int
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    max_steps: int


def load_config(env_file: str = ".env") -> AppConfig:
    env_path = Path(env_file)
    load_dotenv(dotenv_path=env_path if env_path.exists() else None, override=False)

    run_mode = _env_str("RUN_MODE", "FULL_RUN").upper()
    if run_mode not in {"SMALL_RUN", "FULL_RUN"}:
        run_mode = "FULL_RUN"

    wandb_entity = os.getenv("WANDB_ENTITY")
    hf_repo_id = os.getenv("HF_REPO_ID")
    hf_token = os.getenv("HF_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_run_name = os.getenv("WANDB_RUN_NAME")
    experiment_configs = {
        "v1": ExperimentConfig(
            epochs=_env_int("V1_EPOCHS", 3),
            batch_size=_env_int("V1_BATCH_SIZE", 10),
            eval_batch_size=_env_int("V1_EVAL_BATCH_SIZE", 16),
            learning_rate=_env_float("V1_LEARNING_RATE", 2e-5),
            warmup_steps=_env_int("V1_WARMUP_STEPS", 100),
            weight_decay=_env_float("V1_WEIGHT_DECAY", 0.01),
            max_steps=_env_int("V1_MAX_STEPS", -1),
        ),
        "v2": ExperimentConfig(
            epochs=_env_int("V2_EPOCHS", 4),
            batch_size=_env_int("V2_BATCH_SIZE", 16),
            eval_batch_size=_env_int("V2_EVAL_BATCH_SIZE", 16),
            learning_rate=_env_float("V2_LEARNING_RATE", 3e-5),
            warmup_steps=_env_int("V2_WARMUP_STEPS", 150),
            weight_decay=_env_float("V2_WEIGHT_DECAY", 0.01),
            max_steps=_env_int("V2_MAX_STEPS", -1),
        ),
    }
    experiment_version = _env_str("EXPERIMENT_VERSION", "v1").lower()
    if experiment_version not in experiment_configs:
        experiment_version = "v1"
    selected_experiment = experiment_configs[experiment_version]

    return AppConfig(
        run_mode=run_mode,
        seed=_env_int("SEED", 42),
        dataset_name=_env_str("DATASET_NAME", "facebook/anli"),
        train_splits=_env_list("TRAIN_SPLITS", ["train_r1", "train_r2", "train_r3"]),
        test_splits=_env_list("TEST_SPLITS", ["test_r1", "test_r2", "test_r3"]),
        train_max_rows=_env_int("TRAIN_MAX_ROWS", 10000),
        test_max_rows=_env_int("TEST_MAX_ROWS", 2000),
        data_pickle_path=_env_str("DATA_PICKLE_PATH", "./results/anli_text_classification_data.pickle"),
        label_map_path=_env_str("LABEL_MAP_PATH", "./results/id2label.json"),
        model_name=_env_str("MODEL_NAME", "roberta-base"),
        inference_model_name=_env_str("HF_MODEL_NAME", "kamalchaurasia-iitj/mlops-anli-classifier-roberta"),
        inference_fallback_model_name=_env_str("HF_FALLBACK_MODEL_NAME", "roberta-base"),
        experiment_version=experiment_version,
        experiment_configs=experiment_configs,
        max_length=_env_int("MAX_LENGTH", 512),
        output_dir=_env_str("OUTPUT_DIR", "./results/model"),
        logging_dir=_env_str("LOGGING_DIR", "./results/logs"),
        eval_report_path=_env_str("EVAL_REPORT_PATH", "./results/eval_report.json"),
        epochs=selected_experiment.epochs,
        batch_size=selected_experiment.batch_size,
        eval_batch_size=selected_experiment.eval_batch_size,
        learning_rate=selected_experiment.learning_rate,
        warmup_steps=selected_experiment.warmup_steps,
        weight_decay=selected_experiment.weight_decay,
        max_steps=selected_experiment.max_steps,
        enable_wandb=_env_bool("ENABLE_WANDB", False),
        wandb_project=_env_str("WANDB_PROJECT", "MLOps-ANLI-NLI"),
        wandb_entity=wandb_entity.strip() if wandb_entity and wandb_entity.strip() else None,
        wandb_run_name=wandb_run_name.strip() if wandb_run_name and wandb_run_name.strip() else None,
        wandb_log_eval_report=_env_bool("WANDB_LOG_EVAL_REPORT", True),
        wandb_api_key=wandb_api_key.strip() if wandb_api_key and wandb_api_key.strip() else None,
        push_to_hub=_env_bool("PUSH_TO_HUB", False),
        hf_repo_id=hf_repo_id.strip() if hf_repo_id and hf_repo_id.strip() else None,
        hf_token=hf_token.strip() if hf_token and hf_token.strip() else None,
        push_tokenizer=_env_bool("PUSH_TOKENIZER", True),
        update_wandb_summary=_env_bool("UPDATE_WANDB_SUMMARY", True),
    )


def is_small_run(run_mode: str) -> bool:
    return str(run_mode).upper() == "SMALL_RUN"
