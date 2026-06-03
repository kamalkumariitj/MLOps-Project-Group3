# ANLI Natural Language Inference with RoBERTa

**MLOps Assignment 3 — IIT Jodhpur PGD AI Program**
**Authors:** Chaurasia Kamalkumar Lallanprasad, Solanki Bhavik Pravinbhai, Govardhan Kumar, Mahesh Om Prakash Bali

Fine-tunes `roberta-base` on the `facebook/anli` dataset for adversarial natural language inference. This notebook builds premise/hypothesis pairs, fine-tunes a RoBERTa classification head, evaluates on validation/test splits, and logs experiments to Weights & Biases.

**Task:** 3-way NLI classification with labels `entailment`, `neutral`, and `contradiction`.

---

## Notebook Workflow

The notebook `MLOps_Assignment_3_Fine_Tuning_Classification_roberta.ipynb` runs the full pipeline end-to-end:

1. **Load dataset** — `facebook/anli` from Hugging Face Hub, including round-based splits like `train_r1`, `dev_r1`, `train_r2`, `dev_r2`, `train_r3`, `dev_r3`, `test_r1`, and `test_r2`
2. **Data prep** — format premise/hypothesis pairs, map labels to integers, and concatenate round-specific splits for training and evaluation
3. **Baseline** — optional TF-IDF + Logistic Regression baseline using premise/hypothesis text
4. **Load model** — `RobertaTokenizer` + `RobertaForSequenceClassification` from Hugging Face
5. **Fine-tune** — Hugging Face `Trainer` API with W&B experiment tracking (`report_to="wandb"`)
6. **Evaluate** — `trainer.evaluate()` plus detailed classification metrics and misclassification analysis
7. **Publish** — save and optionally push the fine-tuned model and tokenizer to the Hugging Face Hub

---

## Model

**`roberta-base`**

RoBERTa improves upon BERT by removing the Next Sentence Prediction (NSP) objective, training with dynamic masking, using larger mini-batches, and learning from more text. The `roberta-base` checkpoint has 12 layers, hidden size 768, 12 attention heads, and a byte-level BPE vocabulary.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Key packages:** `transformers`, `torch`, `accelerate`, `sentencepiece`, `wandb`, `scikit-learn`, `datasets`, `huggingface_hub`, `pandas`, `numpy`

### 2. Set environment variables

On macOS/Linux:

```bash
export WANDB_API_KEY=<your W&B API key>
export HF_TOKEN=<your Hugging Face token>
```

On Windows PowerShell:

```powershell
$env:WANDB_API_KEY="<your W&B API key>"
$env:HF_TOKEN="<your Hugging Face token>"
```

### 3. Run the notebook

Open and run `MLOps_Assignment_3_Fine_Tuning_Classification_roberta.ipynb` from top to bottom. The first cell includes environment setup and cleanup logic for W&B.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | `roberta-base` |
| Epochs | 3 |
| Train batch size | 10 |
| Eval batch size | 16 |
| Learning rate | 2e-5 |
| Warmup steps | 100 |
| Weight decay | 0.01 |
| Logging steps | 50 |
| Eval / save strategy | epoch |
| Max sequence length | 512 |

---

## Notes

- The notebook uses the Hugging Face `datasets` library to load `facebook/anli`.
- The task is adversarial natural language inference, with label mapping `0 -> entailment`, `1 -> neutral`, `2 -> contradiction`.
- The notebook concatenates round-specific train/dev splits for a single training and validation dataset when appropriate.

---

## Links

- **Dataset:** https://huggingface.co/datasets/facebook/anli
- **Transformers:** https://huggingface.co/docs/transformers/
- **Weights & Biases:** https://wandb.ai/
