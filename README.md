# RoBERTa ANLI Classifier

MLOps Assignment 2 - IIT Jodhpur PGD AI Program

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

Run inference mode:

`python src/main.py --mode inference --input-text "premise: A man is playing guitar hypothesis: A person is making music"`

Inference mode uses the model configured in `src/config.py`:
`kamalchaurasia-iitj/mlops-anli-classifier-roberta`.

## Run with Docker

Build image:

`docker build -t mlops-assignment3-group3 .`

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

## GitHub Actions Pipeline (`mlops-pipeline.yml`)

Workflow file: `.github/workflows/mlops-pipeline.yml`

Triggers:

- Push to `main` or `develop`
- Manual run from Actions tab (`workflow_dispatch`)

Required repository secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
- `HF_TOKEN`
- `WANDB_API_KEY`

What this workflow does:

1. Checks out the repo.
2. Builds Docker image.
3. Logs in to Docker Hub and pushes image tags:
   - `latest`
   - `${{ github.sha }}`
4. Runs the pipeline inside Docker using `src/main.py` in `SMALL_RUN` mode.
5. Uploads evaluation artifacts.

Manual run options in GitHub Actions:

- `push_to_hub` (`true/false`)
- `hf_repo` (Hugging Face repo id)

## Current Pipeline (As Implemented)

1. Load ANLI from Hugging Face Datasets using load_dataset("facebook/anli").
2. Build text-pair inputs in the form: premise: ... hypothesis: ...
3. Prepare train/test sets from ANLI rounds.
4. Run a baseline TF-IDF + Logistic Regression model.
5. Tokenize with RobertaTokenizer and build custom Torch datasets.
6. Fine-tune RobertaForSequenceClassification via Hugging Face Trainer.
7. Evaluate with accuracy, weighted F1, classification report, and save eval_report.json.
8. Optionally log metrics/artifacts to W&B and push tokenizer/model artifacts to Hugging Face Hub.

## Run Modes

The notebook includes a dedicated run-mode key cell near the top.

- RUN_MODE = SMALL_RUN
- RUN_MODE = FULL_RUN

Behavior:

- SMALL_RUN
	- Reduced dataset slices for fast smoke testing.
	- Lightweight training arguments (single-epoch capped steps).
	- W&B initialization/logging/artifact upload is skipped by design.
- FULL_RUN
	- Uses larger dataset configuration.
	- Full training schedule.
	- W&B tracking is enabled.

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
