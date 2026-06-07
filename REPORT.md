# MLOps Assignment 3 — Final Report

**Name:** Chaurasia Kamalkumar Lallanprasad  
**Roll Number:** G25AIT2028  
**Program:** PGD Artificial Intelligence — IIT Jodhpur  
**Date:** June 2026

---

## Project Overview

This project fine-tunes a pre-trained `FacebookAI/roberta-base` model on the ANLI natural language inference dataset. Each example contains a `premise` and a `hypothesis`, and the model predicts one of three labels:
- `entailment`
- `neutral`
- `contradiction`

The notebook implements the full MLOps workflow from dataset preparation through training, evaluation, and optional model publishing.

---

## 1. Dataset and Task

### Dataset used
- `facebook/anli` from Hugging Face Datasets
- Train splits: `train_r1`, `train_r2`, `train_r3`
- Test splits: `test_r1`, `test_r2`, `test_r3`

### Task description
Natural language inference is a sentence-pair classification task. The model must determine whether a hypothesis is:
- entailed by the premise,
- neutral with respect to the premise, or
- contradictory to the premise.

### Input formatting
Each example is converted into a single RoBERTa-friendly text string:
```python
premise: {premise} hypothesis: {hypothesis}
```
Labels are mapped using the ANLI label order: `entailment=0`, `neutral=1`, `contradiction=2`.

---

## 2. Model Selection Rationale

### Why RoBERTa?
`FacebookAI/roberta-base` was selected because it provides a strong, stable baseline for sentence-pair classification tasks and integrates cleanly with Hugging Face `Trainer`.

Key advantages:
- Byte-level BPE tokenization handles text safely.
- No NSP objective, so the model focuses on language understanding.
- Dynamic masking during pretraining improves generalization.
- Widely used in NLP benchmarks and production pipelines.

### Alternatives considered
- `distilbert-base-cased` — smaller and faster, but less capable on difficult inference tasks.
- `microsoft/deberta-v3-small` — promising architecture, but more sensitive to precision and backend compatibility during debugging.

### Training configuration used
| Setting | Value |
|---------|-------|
| Model | `roberta-base` |
| Epochs | 3 |
| Batch size (train) | 10 |
| Batch size (eval) | 16 |
| Learning rate | 2e-5 |
| Warmup steps | 100 |
| Weight decay | 0.01 |
| Max sequence length | 512 |
| Evaluation strategy | `epoch` |
| Experiment tracking | `wandb` (optional) |

---

## 3. Notebook Workflow

1. Environment setup and dependency installation.
2. Load ANLI using `datasets.load_dataset("facebook/anli")`.
3. Convert premise-hypothesis pairs into text strings and map labels.
4. Tokenize inputs with `RobertaTokenizer`, padding and truncating to 512 tokens.
5. Create PyTorch datasets for training and evaluation.
6. Load `RobertaForSequenceClassification` with `num_labels=3`.
7. Train using `Trainer` and evaluate with custom metrics (`accuracy`, weighted F1).
8. Save final evaluation results to `eval_report.json` and optionally upload artifacts to W&B.

---

## 4. Key Findings

- The notebook now accurately reflects the ANLI NLI classification task.
- `gdown` and other required packages must be installed before importing the notebook dependencies.
- The `SMALL_RUN` mode is useful for quick debugging, while full training uses the complete selected ANLI splits.
- Final evaluation output is stored in `eval_report.json`, making results reproducible and easy to inspect.

---

## 5. Challenges and Learnings

### Dependency management
The notebook originally attempted to import optional packages such as `gdown` without guaranteeing installation. A complete `requirements.txt` is necessary for reproducible runs.

### Task-aligned preprocessing
For ANLI, it is essential to format the input as a paired premise-hypothesis string. This ensures RoBERTa receives the correct context for NLI classification.

### Model choice
RoBERTa provided a stable baseline. A more advanced model like DeBERTa-v3 could be attempted later, but RoBERTa avoids several hardware and precision pitfalls while still delivering strong NLI performance.

### Evaluation tracking
Saving `eval_report.json` ensures that metrics such as accuracy, weighted F1, and loss are preserved outside the notebook. This is important for MLOps reproducibility.

---

## 6. Recommendations

1. Install dependencies from `requirements.txt` before running the notebook.
2. Use `RUN_MODE = 'SMALL_RUN'` for initial debugging.
3. Add a separate validation split if the notebook will be extended beyond this assignment.
4. Review `eval_report.json` after training for final performance metrics.
5. Push the fine-tuned model to Hugging Face Hub only after verifying that the evaluation metrics are satisfactory.

---

## Notes

- This report is aligned with the current notebook `MLOps_Assignment_3_Fine_Tuning_Classification_roberta.ipynb`.
- The current notebook is ready for both local and Kaggle execution once dependencies are installed.
- The main output artifacts are `eval_report.json`, model checkpoints saved by `Trainer`, and optional W&B logs.

