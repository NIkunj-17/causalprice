---
title: CausalPrice
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.31.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# CausalPrice

### Causal effect of price on customer engagement — Amazon 2023

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-teal)](https://huggingface.co/spaces/NIKUNJ-17/causalprice)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/NIkunj-17/causalprice)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![EconML](https://img.shields.io/badge/EconML-0.15.1-green)](https://econml.azurewebsites.net/)
[![DoWhy](https://img.shields.io/badge/DoWhy-0.11.1-orange)](https://py-why.github.io/dowhy/)

---

## What this project does

Most ML models answer: *"given these features, what will the outcome be?"*

This project answers a harder question: **"if we intervene on price, what actually changes — after controlling for everything else?"**

Using 258,069 real Amazon Electronics products (Sep 2023), this system estimates the **causal effect** of price on customer engagement (review volume), controlling for 7 confounders using Double ML + Causal Forest. The result is not a correlation — it is a causal estimate with rigorous statistical validation.

**Live demo:** [huggingface.co/spaces/NIKUNJ-17/causalprice](https://huggingface.co/spaces/NIKUNJ-17/causalprice)

---

## Key findings

| Metric | Value |
|--------|-------|
| Population ATE | **−0.7196** log-reviews |
| 95% Confidence Interval | [−1.2134, −0.2258] |
| p-value | **0.0043** (significant) |
| CATE range | −1.94 to +1.43 |
| Rosenbaum Γ | **17.39** |
| Products analysed | 258,069 across 62 categories |
| Holdout validation | 24,237 products, direction correct |

**The headline finding:** Premium pricing causally suppresses customer engagement on average (ATE = −0.72, p = 0.004). But this average hides dramatic heterogeneity — individual product elasticities range from −1.94 to +1.43.

**The insight:** Premium pricing *helps* in experience goods (headphones, video games — quality is hard to assess before purchase, so price signals it) and *hurts* in commodity categories (smart home devices, cables — specs are transparent, high price raises expectations without delivering). This is consistent with **information asymmetry theory** (Akerlof, 1970).

---

## System design

```
Raw data (1.4M Amazon products)
        │
        ▼
Feature engineering (7 confounders)
├── rating z-score within category
├── log review count
├── log category competition
├── price tier (budget/mid/premium)
├── discount depth
├── bestseller flag
└── log absolute price level
        │
        ▼
Causal DAG (DoWhy)
price_pct → log_reviews
confounders → price_pct
confounders → log_reviews
        │
        ▼
Identification & Estimation
├── Linear DML → ATE + 95% CI
└── Causal Forest DML → CATE (heterogeneous effects)
        │
        ▼
Validation
├── 3 DoWhy refutation tests (placebo, random cause, subset)
├── Held-out category split (7 unseen categories)
├── Bootstrap CI (200 resampled models)
└── Rosenbaum sensitivity bounds (Γ = 17.39)
        │
        ▼
Gradio Dashboard (HuggingFace Spaces)
├── Live metric cards (CATE, elasticity, review impact, percentile)
├── Price sensitivity curve with bootstrap CI
├── Competitor scatter map
├── Elasticity distribution
├── CATE heatmap (price × competition)
└── Category ranking
```

---

## Why log(review_count) as outcome?

Binary P(rating ≥ 4.0) has a 90% baseline on Amazon data → Bernoulli variance = 0.09 → near-zero signal.

`log(1 + review_count)` has std ≈ 1.85 → **6× more statistical signal**. Review volume is a published proxy for purchase volume and satisfaction (Chevalier & Mayzlin, 2006).

---

## Technical stack

| Component | Choice |
|-----------|--------|
| Causal identification | DoWhy DAG |
| ATE estimation | LinearDML (EconML) |
| CATE estimation | CausalForestDML (EconML) |
| Nuisance models | GradientBoostingRegressor (sklearn) |
| Cross-fitting | 5-fold |
| Bootstrap CI | 200 resampled Causal Forest models |
| Sensitivity | Rosenbaum bounds |
| Generalization | Held-out category split |
| Frontend | Gradio + Plotly |
| Deployment | HuggingFace Spaces |

---

## Run locally (Windows)

```powershell
git clone https://github.com/NIkunj-17/causalprice.git
cd causalprice
py -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset:
```
https://www.kaggle.com/datasets/asaniczka/amazon-products-dataset-2023-1-4m-products
```
Place `amazon_products.csv` and `amazon_categories.csv` in `data/`.

```powershell
python train.py    # ~45-60 min (includes 200-sample bootstrap)
python app.py      # launches at http://localhost:7860
```

---

## Project structure

```
causalprice/
├── src/
│   ├── features.py        # data loading + feature engineering
│   └── causal_model.py    # DoWhy DAG + DML + Causal Forest + validation
├── data/
│   ├── cate_model.pkl     # trained CausalForestDML
│   ├── results.pkl        # ATE, CI, refutation results
│   ├── bootstrap_curves.pkl  # 200 bootstrap CATE curves
│   ├── features.csv       # processed feature matrix
│   └── category_stats.csv # per-category statistics for inference
├── app.py                 # Gradio dashboard
├── train.py               # end-to-end training script
└── requirements.txt
```

---
