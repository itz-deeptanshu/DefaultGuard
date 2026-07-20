"""
Loan Default / Credit Risk Prediction — End-to-End Pipeline
==============================================================

A single-file, runnable version of the full project:
  1. EDA                  -> understand the imbalance & data quality issues
  2. Preprocessing        -> clean, encode, split
  3. Imbalance handling   -> baseline vs class weights vs SMOTE
  4. Threshold tuning     -> cost-based decision threshold (FN vs FP tradeoff)

Dataset: Credit Risk Dataset (Kaggle) - 32,581 loan applicants, 12 features,
binary target `loan_status` (1 = default). ~78% repaid / ~22% default.

Usage:
    python3 loan_default_pipeline.py --data path/to/credit_risk_dataset.csv

If --data is omitted, it looks for data/credit_risk_dataset.csv relative to
the current directory.

Outputs (written to ./outputs/):
    figures/            EDA and threshold-tuning charts
    models/             saved model, scaler, comparison tables, final report
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


# ==============================================================
# PHASE 1: EDA
# ==============================================================
def run_eda(df: pd.DataFrame, fig_dir: str) -> None:
    print("\n" + "=" * 70)
    print("PHASE 1: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    print("Shape:", df.shape)

    missing = df.isnull().sum()
    print("\nMissing values:\n", missing[missing > 0])

    counts = df["loan_status"].value_counts()
    pct = df["loan_status"].value_counts(normalize=True) * 100
    print(f"\nRepaid (0):  {counts[0]:>6} ({pct[0]:.1f}%)")
    print(f"Default (1): {counts[1]:>6} ({pct[1]:.1f}%)")
    print(f"Imbalance ratio ~ {counts[0] / counts[1]:.2f} : 1")
    print("\nA model that always predicts 'repaid' would score "
          f"{pct[0]:.1f}% accuracy while catching ZERO defaulters. "
          "Accuracy is the wrong metric here.")

    # Sanity checks (this dataset has known dirty rows)
    n_old = (df["person_age"] > 100).sum()
    n_bad_emp = (df["person_emp_length"] > df["person_age"]).sum()
    print(f"\nRows with age > 100: {n_old}")
    print(f"Rows with employment length exceeding age: {n_bad_emp}")

    # -- Figure 1: class imbalance
    fig, ax = plt.subplots(figsize=(6, 5))
    counts.plot(kind="bar", color=["#4C72B0", "#C44E52"], ax=ax)
    ax.set_xticklabels(["Repaid (0)", "Default (1)"], rotation=0)
    ax.set_title("Class Imbalance in Loan Status")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts):
        ax.text(i, v + 300, f"{v}\n({pct[i]:.1f}%)", ha="center")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/01_target_imbalance.png", dpi=120)
    plt.close()

    # -- Figure 2: numeric features by class
    numeric_cols = ["person_age", "person_income", "person_emp_length",
                     "loan_amnt", "loan_int_rate", "loan_percent_income",
                     "cb_person_cred_hist_length"]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.boxplot(data=df, x="loan_status", y=col, hue="loan_status",
                    ax=axes[i], palette=["#4C72B0", "#C44E52"], legend=False)
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["Repaid", "Default"])
        axes[i].set_title(col)
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle("Numeric Features vs Loan Status", y=1.01, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/02_numeric_by_class.png", dpi=120, bbox_inches="tight")
    plt.close()

    # -- Figure 3: categorical default rates
    cat_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        rate = df.groupby(col)["loan_status"].mean().sort_values(ascending=False) * 100
        rate.plot(kind="bar", ax=axes[i], color="#C44E52")
        axes[i].set_title(f"Default Rate by {col}")
        axes[i].set_ylabel("Default rate (%)")
        axes[i].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/03_categorical_default_rate.png", dpi=120)
    plt.close()

    print(f"\nEDA figures saved to {fig_dir}/")


# ==============================================================
# PHASE 2: PREPROCESSING
# ==============================================================
def run_preprocessing(df: pd.DataFrame, model_dir: str):
    print("\n" + "=" * 70)
    print("PHASE 2: PREPROCESSING")
    print("=" * 70)

    before = len(df)
    df = df[df["person_age"] <= 100]
    df = df[df["person_emp_length"] <= df["person_age"]]
    print(f"Dropped {before - len(df)} rows with impossible age/employment values")

    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())
    df["loan_int_rate"] = df.groupby("loan_grade")["loan_int_rate"].transform(
        lambda x: x.fillna(x.median())
    )
    print("Remaining missing values:", df.isnull().sum().sum())

    grade_map = {g: i for i, g in enumerate(sorted(df["loan_grade"].unique()))}
    df["loan_grade_encoded"] = df["loan_grade"].map(grade_map)
    df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map({"Y": 1, "N": 0})
    df = pd.get_dummies(df, columns=["person_home_ownership", "loan_intent"], drop_first=True)
    df = df.drop(columns=["loan_grade"])

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train shape: {X_train.shape}, default rate: {y_train.mean():.3f}")
    print(f"Test shape:  {X_test.shape}, default rate: {y_test.mean():.3f}")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    joblib.dump(scaler, f"{model_dir}/scaler.pkl")
    print(f"Scaler saved to {model_dir}/scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test


# ==============================================================
# PHASE 3: IMBALANCE HANDLING
# ==============================================================
def run_imbalance_comparison(X_train, X_test, y_train, y_test, model_dir: str):
    print("\n" + "=" * 70)
    print("PHASE 3: HANDLING IMBALANCE — BASELINE vs CLASS WEIGHTS vs SMOTE")
    print("=" * 70)

    results = []

    def evaluate(name, model):
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        row = {
            "strategy": name,
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "recall_default": round(recall_score(y_test, pred), 4),
            "precision_default": round(precision_score(y_test, pred), 4),
            "f1_default": round(f1_score(y_test, pred), 4),
        }
        results.append(row)
        print(f"{name:45s} | ROC-AUC {row['roc_auc']:.3f} | "
              f"Recall {row['recall_default']:.3f} | Precision {row['precision_default']:.3f}")
        return proba

    # A) Baseline
    baseline = XGBClassifier(random_state=42, eval_metric="logloss")
    baseline.fit(X_train, y_train)
    evaluate("A) Baseline (no correction) - XGBoost", baseline)

    # B) Class weights
    lr_weighted = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr_weighted.fit(X_train, y_train)
    evaluate("B1) Class Weights - Logistic Regression", lr_weighted)

    rf_weighted = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                          max_depth=12, random_state=42, n_jobs=-1)
    rf_weighted.fit(X_train, y_train)
    evaluate("B2) Class Weights - Random Forest", rf_weighted)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_weighted = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric="logloss")
    xgb_weighted.fit(X_train, y_train)
    evaluate("B3) Class Weights - XGBoost", xgb_weighted)

    # C) SMOTE (train set only)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"\nAfter SMOTE, train default rate: {y_train_smote.mean():.3f}")

    lr_smote = LogisticRegression(max_iter=1000, random_state=42)
    lr_smote.fit(X_train_smote, y_train_smote)
    evaluate("C1) SMOTE - Logistic Regression", lr_smote)

    rf_smote = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    rf_smote.fit(X_train_smote, y_train_smote)
    evaluate("C2) SMOTE - Random Forest", rf_smote)

    xgb_smote = XGBClassifier(random_state=42, eval_metric="logloss")
    xgb_smote.fit(X_train_smote, y_train_smote)
    evaluate("C3) SMOTE - XGBoost", xgb_smote)

    results_df = pd.DataFrame(results).sort_values("recall_default", ascending=False)
    print("\n" + "-" * 70)
    print("STRATEGY COMPARISON (sorted by recall on defaulters)")
    print("-" * 70)
    print(results_df.to_string(index=False))
    results_df.to_csv(f"{model_dir}/strategy_comparison.csv", index=False)

    # Class-weighted XGBoost is carried forward as the production model
    joblib.dump(xgb_weighted, f"{model_dir}/xgb_class_weighted.pkl")
    print(f"\nBest model (class-weighted XGBoost) saved to {model_dir}/xgb_class_weighted.pkl")

    return xgb_weighted


# ==============================================================
# PHASE 4: THRESHOLD TUNING (business cost tradeoff)
# ==============================================================
def run_threshold_tuning(model, X_test, y_test, fig_dir: str, model_dir: str,
                          cost_fn: float = 5.0, cost_fp: float = 1.0):
    print("\n" + "=" * 70)
    print("PHASE 4: THRESHOLD TUNING — FALSE NEGATIVE vs FALSE POSITIVE COST")
    print("=" * 70)
    print(f"Cost assumption: missing a defaulter (FN) = {cost_fn}x, "
          f"wrongly rejecting a good customer (FP) = {cost_fp}x")
    print("Rationale: a missed defaulter costs the bank the FULL loan principal;")
    print("a wrongly-rejected good customer only costs the foregone INTEREST MARGIN.")

    proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 19)
    records = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        total_cost = fn * cost_fn + fp * cost_fp
        recall = tp / (tp + fn) if (tp + fn) else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        records.append({"threshold": round(t, 2), "fn": fn, "fp": fp, "tp": tp, "tn": tn,
                         "total_cost": total_cost, "recall": round(recall, 3),
                         "precision": round(precision, 3)})

    cost_df = pd.DataFrame(records)
    best_row = cost_df.loc[cost_df["total_cost"].idxmin()]
    default_row = cost_df.iloc[(cost_df["threshold"] - 0.5).abs().idxmin()]

    print("\n" + cost_df.to_string(index=False))
    print(f"\nOptimal threshold: {best_row['threshold']} "
          f"(FN={int(best_row['fn'])}, FP={int(best_row['fp'])}, "
          f"Recall={best_row['recall']}, Precision={best_row['precision']}, "
          f"Cost={best_row['total_cost']})")
    print(f"Default threshold 0.5: FN={int(default_row['fn'])}, FP={int(default_row['fp'])}, "
          f"Cost={default_row['total_cost']}")
    savings = default_row["total_cost"] - best_row["total_cost"]
    print(f"Switching thresholds saves {savings:.0f} cost units "
          f"({savings / default_row['total_cost'] * 100:.1f}% reduction).")

    # -- Figure: cost & precision/recall vs threshold
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(cost_df["threshold"], cost_df["total_cost"], marker="o", color="#C44E52")
    axes[0].axvline(best_row["threshold"], color="green", linestyle="--", label=f"Optimal = {best_row['threshold']}")
    axes[0].axvline(0.5, color="gray", linestyle=":", label="Default = 0.5")
    axes[0].set_xlabel("Decision threshold")
    axes[0].set_ylabel(f"Total cost (FN={cost_fn}x, FP={cost_fp}x)")
    axes[0].set_title("Business Cost vs Threshold")
    axes[0].legend()

    axes[1].plot(cost_df["threshold"], cost_df["recall"], marker="o", label="Recall (catch defaulters)")
    axes[1].plot(cost_df["threshold"], cost_df["precision"], marker="s", label="Precision (avoid false alarms)")
    axes[1].axvline(best_row["threshold"], color="green", linestyle="--")
    axes[1].axvline(0.5, color="gray", linestyle=":")
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Precision / Recall Tradeoff")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/04_threshold_tuning.png", dpi=120)
    plt.close()

    # -- Figure: confusion matrices side by side
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, t, label in zip(axes, [0.5, best_row["threshold"]],
                             ["Default (0.5)", f"Tuned ({best_row['threshold']})"]):
        pred = (proba >= t).astype(int)
        cm = confusion_matrix(y_test, pred)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred\nRepaid", "Pred\nDefault"])
        ax.set_yticklabels(["Actual\nRepaid", "Actual\nDefault"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        ax.set_title(f"Threshold = {label}")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/05_confusion_matrix_comparison.png", dpi=120)
    plt.close()

    final_pred = (proba >= best_row["threshold"]).astype(int)
    report = classification_report(y_test, final_pred, target_names=["Repaid", "Default"])
    print("\nFINAL CLASSIFICATION REPORT AT TUNED THRESHOLD:\n", report)

    with open(f"{model_dir}/final_report.txt", "w") as f:
        f.write(f"Optimal threshold: {best_row['threshold']}\n\n{report}")
    cost_df.to_csv(f"{model_dir}/threshold_cost_analysis.csv", index=False)
    print(f"Final report and threshold analysis saved to {model_dir}/")


# ==============================================================
# MAIN
# ==============================================================
def main():
    parser = argparse.ArgumentParser(description="Loan default prediction end-to-end pipeline")
    parser.add_argument("--data", default="data/credit_risk_dataset.csv",
                         help="Path to credit_risk_dataset.csv")
    parser.add_argument("--cost-fn", type=float, default=5.0,
                         help="Relative cost of missing a defaulter (false negative)")
    parser.add_argument("--cost-fp", type=float, default=1.0,
                         help="Relative cost of wrongly rejecting a good customer (false positive)")
    args = parser.parse_args()

    fig_dir = "outputs/figures"
    model_dir = "outputs/models"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(args.data)

    run_eda(df.copy(), fig_dir)
    X_train, X_test, y_train, y_test = run_preprocessing(df.copy(), model_dir)
    best_model = run_imbalance_comparison(X_train, X_test, y_train, y_test, model_dir)
    run_threshold_tuning(best_model, X_test, y_test, fig_dir, model_dir,
                          cost_fn=args.cost_fn, cost_fp=args.cost_fp)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE. See outputs/figures/ and outputs/models/")
    print("=" * 70)


if __name__ == "__main__":
    main()
