# RoBERTa ANLI Classifier

MLOps Assignment 3 - IIT Jodhpur PGD AI Program

## Authors

| Roll Number | Name |
|---|---|
| G25AIT2028 | Chaurasia Kamalkumar Lallanprasad |
| G25AIT2106 | Solanki Bhavik Pravinbhai |
| G25AIT2035 | Govardhan Kumar |
| G25AIT2057 | Mahesh Om Prakash Bali |

This project fine-tunes roberta-base on facebook/anli for 3-class natural language inference: entailment, neutral, contradiction.

## Notebook

Primary notebook:

- MLOps_Assignment_3_Fine_Tuning_Classification_roberta_V2.ipynb

## Python Program (Script Pipeline)

This repository also includes a Python script-based pipeline for non-notebook runs:

- `src/main.py`: orchestrates pipeline stages (`data -> train -> eval`) and inference mode
- `src/data.py`: loads `facebook/anli`, cleans data, encodes labels, saves processed dataset
- `src/train.py`: tokenizes text pairs, fine-tunes `roberta-base`, saves model/tokenizer/metrics
- `src/eval.py`: evaluates saved model and writes `eval_report.json`
- `src/inference.py`: runs single-text inference against the configured Hugging Face model
- `src/config.py`: central configuration loader from environment variables / `.env`
- `src/utils.py`: shared dataset, cleaning, metrics, and seed utilities

### Run the Python Pipeline

1. Install dependencies:

`pip install -r requirements.txt`

2. To run locally create `.env` for secrets (for example `WANDB_API_KEY`, `HF_TOKEN`).

4. Run all stages:

`python src/main.py --run-mode FULL_RUN --stage all`

Quick smoke run:

`python src/main.py --run-mode SMALL_RUN --stage all --disable-wandb --no-push-to-hub`

Run only one stage:

- `python src/main.py --stage data`
- `python src/main.py --stage train`
- `python src/main.py --stage eval`

## Run inference mode:

`python src/main.py --mode inference --input-text "premise: A man is playing guitar hypothesis: A person is making music"`

Optional (inference accepts it, but does not use it): `--run-mode SMALL_RUN|FULL_RUN`

Inference mode uses the model configured in `src/config.py`:
`kamalchaurasia-iitj/mlops-anli-classifier-roberta`.

## Run with Docker

Build image:

`docker build --build-arg HF_MODEL_NAME=kamalchaurasia-iitj/mlops-anli-classifier-roberta -t mlops-assignment3-group3 .`

Build image with a custom Hugging Face model:

`docker build --build-arg HF_MODEL_NAME=your-username/your-model -t mlops-assignment3-group3 .`

Default HF model is `kamalchaurasia-iitj/mlops-anli-classifier-roberta`, and fallback model is `roberta-base`.

Run Docker in inference mode:

`docker run --rm -e APP_MODE=inference -e INPUT_TEXT="premise: A man is playing guitar hypothesis: A person is making music" mlops-assignment3-group3`

Default Docker mode is also `inference`, so `APP_MODE=inference` is optional.

Run Docker in train mode:

`docker run --rm -e APP_MODE=train mlops-assignment3-group3`

Run Docker in train mode with explicit run mode:

`docker run --rm -e APP_MODE=train -e RUN_MODE=SMALL_RUN mlops-assignment3-group3`

Run full command (same pattern used in CI) and persist outputs locally:

```bash
mkdir -p results
docker run --rm \
  --user root \
  -v "$(pwd):/app" \
  mlops-assignment3-group3 python src/main.py \
    --run-mode SMALL_RUN \
    --disable-wandb \
    --no-push-to-hub \
    --data-path /app/results/anli_text_classification_data.pickle \
    --label-map-path /app/id2label.json \
    --output-dir /app/results/model \
    --logging-dir /app/results/logs \
    --report-path /app/results/eval_report.json
```

## GitHub Actions CI (`ci.yml`)

Workflow file: `.github/workflows/ci.yml`

Triggers:

- Push to `develop`
- Pull request targeting `main`

What this workflow does:

1. Checks out the repository.
2. Sets up Python `3.11`.
3. Installs dependencies and `flake8`.
4. Runs lint checks on `src/` with max line length `120`.

## GitHub Actions Inference Pipeline (`inference.yml`)

Workflow file: `.github/workflows/inference.yml`

Trigger:

- Manual run only (`workflow_dispatch`)

Manual run inputs:

- `input_text` (text for inference)
- `hf_model_name` (HF model name used as Docker build arg, default `kamalchaurasia-iitj/mlops-anli-classifier-roberta`)

Required repository secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
- `HF_TOKEN`

What this workflow does:

1. Checks out the repository.
2. Builds Docker image with `HF_MODEL_NAME`.
3. Logs in and pushes Docker image to Docker Hub.
4. Runs Docker in inference mode using the provided `input_text`.

## Current Pipeline (As Implemented)

1. Trigger `inference.yml` manually from GitHub Actions (`workflow_dispatch`).
2. Provide `input_text` and optionally `hf_model_name` (default: `kamalchaurasia-iitj/mlops-anli-classifier-roberta`).
3. Build Docker image using `HF_MODEL_NAME` build argument.
4. Log in to Docker Hub and push image tags (`latest` and `${{ github.sha }}`).
5. Run Docker container in inference mode (`APP_MODE=inference`) with `INPUT_TEXT` and `HF_TOKEN`.
6. Produce inference output as JSON with predicted label and class scores.

## Run Modes

The notebook includes a dedicated run-mode key cell near the top.

- RUN_MODE = SMALL_RUN
- RUN_MODE = FULL_RUN

Behavior: (Optional for inference mode, useful only for train mode)

- SMALL_RUN
	- Reduced dataset slices for fast smoke testing.
	- Lightweight training arguments (single-epoch capped steps).
	- W&B initialization/logging/artifact upload is skipped by design.
- FULL_RUN
	- Uses larger dataset configuration.
	- Full training schedule.
	- W&B tracking is enabled.

Script run modes:

- `--mode train` (default): runs data/train/eval flow, use `--run-mode SMALL_RUN|FULL_RUN`.
- `--mode inference`: runs single-text inference flow.
- In inference flow, `--run-mode` is optional and effectively ignored.

## Key Configuration Used In Notebook

Model stack:

- HF model class: RobertaForSequenceClassification
- Tokenizer: RobertaTokenizer
- Base checkpoint: roberta-base
- Max sequence length: 512

Training arguments are mode-dependent:

- SMALL_RUN
	- num_train_epochs: 1
	- max_steps: 80
	- per_device_train_batch_size: 8
	- per_device_eval_batch_size: 16
	- learning_rate: 2e-5
	- logging_steps: 10
	- eval_strategy: steps
	- eval_steps: 20
	- save_strategy: no
	- report_to: none
- FULL_RUN
	- num_train_epochs: 3
	- per_device_train_batch_size: 10
	- per_device_eval_batch_size: 16
	- learning_rate: 2e-5
	- warmup_steps: 100
	- weight_decay: 0.01
	- logging_steps: 50
	- eval_strategy: epoch
	- save_strategy: epoch
	- report_to: wandb

## Setup

1. Install dependencies:

pip install -r requirements.txt

2. Optional environment variables:

- WANDB_API_KEY for Weights & Biases logging
- HF_TOKEN for Hugging Face Hub push

3. Run notebook top to bottom.

Important:

- If you change RUN_MODE, re-run from the run-mode key cell onward.
- If packages are newly installed in notebook kernel, re-run earlier setup cells after kernel restart.

## Outputs

Generated by notebook/script execution:

- `results/anli_text_classification_data.pickle`
- `results/id2label.json`
- `results/model/` (trained model + tokenizer + train metrics)
- `results/eval_report.json`
- `results/logs/` (training logs)

## Hugging Face and W&B Notes

- The push cell currently pushes tokenizer by default; model push line is present and can be uncommented.
- W&B summary update in the push cell is guarded so it does not fail when no active run exists.

## Links

- **Github:** [GitHub] (https://github.com/kamalkumariitj/MLOps-Project-Group3)
- **Kaggle notebook:** https://www.kaggle.com/code/kamalkumarg25ait2028/assignment3
- **Hugging Face model:** [kamalchaurasia-iitj/mlops-anli-classifier-roberta](https://huggingface.co/kamalchaurasia-iitj/mlops-anli-classifier-roberta)
- **Docker Image (Public):** https://hub.docker.com/r/kamalchaurasia/mlops-assignment3-group3
- **W&B project dashboard:** https://api.wandb.ai/links/kamalchaurasia-iit-jodhpur/heos7wc2
- **Dataset:** [Facebook/anli](https://huggingface.co/datasets/facebook/anli)
