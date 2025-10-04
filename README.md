# Movie-Genre-NLP — From Baseline to Transformers

Baseline → FFNN → RNN → Transformers for classifying movie genres from plot summaries.  
Includes clean EDA, reproducible splits, MLflow tracking, and handy viz (curves, confusion matrices, PR/ROC).

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/HF-Transformers-ffcc00)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-00b3b3)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

---

## TL;DR
We benchmark text representations and model families for movie-genre classification (27 classes):
- **FFNN** with **TF‑IDF+SVD** and **Sentence-Transformers (MiniLM)** embeddings.
- **RNNs**: BiGRU / BiLSTM with pretrained Word2Vec/FastText initializations.
- **Lightweight Transformers**: DistilRoBERTa, ELECTRA-base, DistilBERT, ALBERT.

We log metrics and artifacts to MLflow and export plots (loss/metric curves, CM, PR/ROC).
A short latency probe demonstrates fast inference for the best Transformer on 8GB GPUs.

---

## Data
Two CSVs with movie plot descriptions:
- `train_*.csv` — columns: `text`, `genre` (27 classes)
- `test_*.csv`  — columns: `text` (predict `genre`)

Splits: **80/10/10** (train/val/hold), stratified by label.  
Preprocessing produces two views:
- `classic_text` — spaCy lemmatization + POS filter (NOUN, PROPN, VERB, ADJ, ADV) + Gensim bigrams.
- `neural_text`  — minimal clean for Transformers.

Artifacts are saved under `artifacts/`:
- `artifacts/reports/` — plots and CSV summaries
- `artifacts/models/` — best checkpoints
- `artifacts/mlflow_artifacts/` — MLflow local store

---

## Setup

> **Python 3.10+** recommended.

```bash
# 1) create venv (example)
python -m venv .venv
source .venv/bin/activate     # on Windows: .venv\Scripts\activate

# 2) install deps
pip install -r requirements.txt

# 3) install PyTorch to match your CUDA (choose one from https://pytorch.org/get-started/locally/)
# Example (CUDA 12.x):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4) spaCy small English model
python -m spacy download en_core_web_sm
```

> If you prefer pinning the spaCy model via `pip`, you can use:
> `pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl`

---

## How to run

Open **`notebook.ipynb`** and run all cells in order. The notebook:
1. Loads data, runs EDA, and prepares `classic_text` / `neural_text`.
2. Trains **FFNN** on different embeddings (BoW/TF‑IDF/Word2Vec/FastText/MiniLM).
3. Trains **RNNs** (BiGRU/BiLSTM) with/without pretrained embeddings.
4. Fine-tunes **Transformers** (DistilRoBERTa, ELECTRA‑base, DistilBERT, ALBERT).
5. Logs artifacts to **MLflow** (`artifacts/mlflow_artifacts/`).
6. Summarizes results and renders:
   - `evolution_bar_val_macroF1.png` — best-of-family bar chart
   - `*_cm_val.png` — confusion matrices
   - `*_epoch_metrics.csv` — training curves
   - optional latency numbers for the best Transformer

---

## Results (validation, macro‑F1)

| Family        | Model (example run)                             | Macro‑F1 | Acc  |
|---------------|--------------------------------------------------|---------:|-----:|
| Transformer   | DistilRoBERTa (`trf_distilroberta-base…`)       | **0.474**| 0.640|
| Transformer   | ELECTRA‑base (`trf_electra-base-discriminator…`)| 0.472    | 0.645|
| Transformer   | DistilBERT / ALBERT‑base                        | 0.466/0.452 | – |
| FFNN          | TF‑IDF+SVD (512)                                 | 0.383    | 0.546|
| FFNN          | MiniLM (Sentence-Transformers)                   | ~0.389   | 0.540|
| RNN           | BiGRU (W2V trainable)                            | 0.352    | 0.540|

**Takeaway.** Lightweight Transformers reliably outperform FFNN and RNN while staying fast at inference.

---

## Inference latency (GPU 8GB, DistilRoBERTa)
Measured at `max_length=192`, `batch_size=32`:

- ~**421 samples/s** (**~2.4 ms/sample**)

> See the latency cell in the notebook for exact numbers and settings.

---

## Reproducibility
- MLflow is configured to use a local folder:
  ```python
  mlflow.set_tracking_uri((ART/"mlflow_artifacts").as_uri())
  mlflow.set_experiment("movie-genre-nlp")
  ```
- All artifacts (curves/plots/CSVs) are stored in `artifacts/reports/`.
- Best checkpoints are saved in `artifacts/models/` (not tracked by git by default).

---

## Future work
- Light hyperparameter sweep for Transformers (`max_length`, `lr`, label smoothing).
- Focal loss / refined class-weights for rare genres.
- TF‑IDF‑weighted averaging or SIF for W2V/FT baselines.
- Optional post‑processing rules for “format” genres (e.g., `short`, `documentary`).

---

## License
MIT (add `LICENSE` file if you plan to distribute).
