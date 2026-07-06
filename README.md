# Aspect-Based Sentiment Analysis (ABSA) Experiments

Experiments in aspect-based sentiment analysis: fine-tuning and benchmarking transformer models (DeBERTa-v3, mT5, SetFit) on public ABSA datasets, plus a preprocessing pipeline that turns scraped think-tank articles (Brookings, Hudson Institute) into sentence-level data ready for annotation in [brat](https://brat.nlplab.org/).

## What's here

### Model fine-tuning & evaluation

| Script | Description |
|--------|-------------|
| `deberta_finetune.py` | Fine-tunes `yangheng/deberta-v3-base-absa-v1.1` on the [M-ABSA](https://github.com/swaggy66/M-ABSA) multilingual dataset (aspect + sentence → polarity classification). Runs on Apple Silicon (MPS) or CPU. |
| `deberta_finetune_testing.py` | Evaluates the fine-tuned DeBERTa checkpoint. |
| `deberta_acl.py`, `deberta_sem.py` | DeBERTa evaluation on the ACL-14 Twitter and SemEval (Laptops) benchmarks. |
| `test_mabsa.py` | Head-to-head comparison of DeBERTa vs SetFit ABSA models on M-ABSA data. |
| `test_brat_deberta.py`, `test_brat_mt5.py`, `test_brat_setfit.py` | Runs each model over brat-annotated sentences and scores against the gold annotations. |
| `testing_models/` | Minimal usage examples for the off-the-shelf DeBERTa and SetFit ABSA models. |

### Data preparation

| Script | Description |
|--------|-------------|
| `crawler/` | Scrapers for Brookings and Hudson Institute article listings and full text, plus data-cleaning utilities. |
| `Hudson_prepro.py` | Cleans scraped article text and splits it into one sentence per line for annotation. |
| `compress.py`, `modify.py` | Utility scripts for reshaping annotation files. |
| `brat_format/` | Sentence text + `.ann` annotation files in brat standoff format. |

### Datasets

- `acl_14.txt` — ACL-14 Twitter ABSA benchmark
- `Laptops Train.xml` — SemEval Laptops ABSA benchmark
- `test.txt`, `test_hu.txt` — held-out sentence sets

> **Note:** The raw scraped article CSVs (`Final_Brooking.csv`, `Final_Hudson.csv`) are not included in this repository — the preprocessing scripts expect them in the project root. Model checkpoints are likewise not committed; scripts download base models from the Hugging Face Hub.

## Setup

```bash
pip install -r requirements.txt
```

Scripts auto-detect Apple Silicon (`mps`) and fall back to CPU.

`test_brat_mt5.py` additionally expects the [M-ABSA](https://github.com/swaggy66/M-ABSA) repository checked out as `M-ABSA-main/` in the project root (it imports `T5FineTuner` from `M-ABSA-main/eval_baseline_mT5/main.py`) plus a fine-tuned mT5 checkpoint under `M-ABSA-main/eval_baseline_mT5/outputs/`.
