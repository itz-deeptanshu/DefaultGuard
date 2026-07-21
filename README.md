# Loan Default / Credit Risk Prediction

An end-to-end ML pipeline that predicts whether a loan applicant will
default, built around two core problems:

1. **Imbalanced data** — most borrowers repay, so a naive model just
   predicts "repaid" for everyone and looks accurate while being useless
2. **Asymmetric business cost** — missing a real defaulter is far more
   expensive than wrongly rejecting a good customer, so the decision
   threshold should reflect that, not sit at a default 0.5

## Files

| File | Purpose |
|---|---|
| `loan_default_pipeline.py` | Single-file, runnable pipeline (EDA → preprocessing → imbalance handling → threshold tuning) |
| `data/credit_risk_dataset.csv` | Input data (place here, or point `--data` at it) |
| `outputs/figures/` | All generated charts |
| `outputs/models/` | Saved model, scaler, comparison tables, final report |

## How to run

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn

python3 loan_default_pipeline.py --data data/credit_risk_dataset.csv
```

Optional flags to set your own cost ratio for threshold tuning:

```bash
python3 loan_default_pipeline.py --data data/credit_risk_dataset.csv --cost-fn 5 --cost-fp 1
```

`--cost-fn` / `--cost-fp` are relative weights for the two error types
(see "Threshold tuning" below) — replace with real numbers
(e.g. average loan principal ÷ average interest margin) once available

## Dataset

[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
— 32,581 loan applicants, 12 features (age, income, employment length, home
ownership, loan intent/grade/amount/interest rate, credit history length,
prior default flag), binary target `loan_status` (1 = default).

Class balance: **78.2% repaid / 21.8% default** (~3.6 : 1 imbalance).

## What each phase does

### Phase 1 — EDA
- Confirms the class imbalance and shows why accuracy is a misleading
  metric here (a "predict repaid always" model scores 78% accuracy while
  catching zero defaulters).
- Flags and removes 902 rows with impossible values (age > 100, employment
  length exceeding age — data entry errors, not real people).
- Charts feature distributions and default rates by category.

### Phase 2 — Preprocessing
- Imputes missing `person_emp_length` (median) and `loan_int_rate`
  (median within each loan grade).
- Encodes `loan_grade` ordinally (A→G is a real risk ordering), one-hot
  encodes nominal categoricals (home ownership, loan intent).
- Stratified 80/20 train/test split — both sets keep the same 21.5%
  default rate. Scaler fit on train only, to avoid leakage into test.

### Phase 3 — Imbalance handling
Three strategies compared, each with Logistic Regression, Random Forest,
and XGBoost:

| Strategy | Best model | ROC-AUC | Recall (default) | Precision (default) |
|---|---|---|---|---|
| Baseline (no fix) | XGBoost | 0.951 | 0.752 | 0.957 |
| **Class weights** | **XGBoost** | **0.953** | **0.802** | 0.842 |
| SMOTE | XGBoost | 0.950 | 0.749 | 0.963 |

**Class-weighted XGBoost wins** — using `scale_pos_weight` to penalize
missed defaulters during training, it catches more real defaulters (80%
vs 75% recall) than SMOTE oversampling, at a similar ROC-AUC. This model is
carried forward to threshold tuning.

### Phase 4 — Threshold tuning (the FN vs FP tradeoff)

The two mistakes a credit model can make are **not equally costly**:

- **False Negative** (predicted "repay," actually defaults): the bank loses
  the **entire loan principal**.
- **False Positive** (predicted "default," actually would repay): the bank
  only loses the **interest profit margin** on that one loan — the
  applicant can reapply or be sent for manual review.

Because a missed defaulter is much more expensive, the default 0.5
threshold is the wrong choice. Using an illustrative 5:1 cost ratio
(`--cost-fn 5 --cost-fp 1`), the cost-minimizing threshold comes out to
**0.30**, not 0.50:

| Threshold | FN | FP | Recall | Precision | Total cost |
|---|---|---|---|---|---|
| 0.50 (default) | 271 | 205 | 0.80 | 0.84 | 1560 |
| **0.30 (tuned)** | **161** | **649** | **0.88** | **0.65** | **1454** |

Lowering the threshold catches ~40% more defaulters (161 vs 271 missed) at
the cost of more false alarms on good customers — worth it, since one
missed default wipes out the profit from many good loans.

> Replace `--cost-fn` / `--cost-fp` with real numbers once available —
> e.g. average loan principal ÷ average interest margin per loan — rather
> than the illustrative 5:1 used here.

## Final model performance

XGBoost + class weights, decision threshold = 0.30:

```
              precision    recall  f1-score   support
      Repaid       0.96      0.87      0.91      4971
     Default       0.65      0.88      0.75      1365
    accuracy                           0.87      6336
```

## Key takeaways

1. Accuracy is misleading on imbalanced data — recall/precision on the
   minority (default) class is what matters.
2. Class weighting beat SMOTE on this dataset — worth comparing both
   rather than assuming SMOTE is always the right fix.
3. The "right" decision threshold depends on business costs, not a
   default 0.5 — in lending, catching defaulters is worth more than
   avoiding false alarms.
