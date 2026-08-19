"""Classifier suite prognosis baseline using annotations and tooth metadata.

Evaluates multiple scikit-learn models on binary features and outputs a summary
table comparing Validation and Test ROC AUC, Balanced Accuracy, and Accuracy.

Example:
    python evaluate_classifiers.py \
        --annotation-csv ml_ready_cbct_annotation.csv \
        --outcome-xlsx "Dataset A - Overview.xlsx" \
        --split-csv checkpoints/prelim_split_seed42.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

ANNOTATION_COLUMNS = [
    "Full-coverage Restoration",
    "Presence of Proximal Teeth",
    "Coronal Defect",
    "Root Rest > 1:1",
    "Canal Visibility",
    "Previous Root Filling",
    "Periapical Lesion",
]


def normalize_case_id(value):
    match = re.search(r"DSA[-_ ]?0*(\d+)", str(value), re.IGNORECASE)
    if match is None:
        return None
    return f"DSA{int(match.group(1)):03d}"


def extract_pai(value):
    match = re.search(r"\d+", str(value))
    return None if match is None else int(match.group())


def prognosis_label(row):
    post_raw = str(row["Follow-up CBCT-PAI [POST]"]).strip().lower()
    if post_raw in {"", "-", "nan", "none"} or "extract" in post_raw:
        return None

    pre = extract_pai(row["Pre-op CBCT-PAI [PRE]"])
    post = extract_pai(row["Follow-up CBCT-PAI [POST]"])
    if post is None:
        return None

    # Binary prognosis target: healed=0; healing/non-healed=1.
    if post <= 2:
        return 0
    if pre is None:
        return None
    return 1 if post < pre else 1


def tooth_features(tooth_number):
    tooth_number = int(tooth_number)
    if not 1 <= tooth_number <= 32:
        raise ValueError(f"Invalid Universal tooth number: {tooth_number}")

    anterior = {6, 7, 8, 9, 10, 11, 22, 23, 24, 25, 26, 27}
    premolar = {4, 5, 12, 13, 20, 21, 28, 29}
    return {
        "mandibular": float(tooth_number >= 17),
        "anterior": float(tooth_number in anterior),
        "premolar": float(tooth_number in premolar),
        "molar": float(tooth_number not in anterior and tooth_number not in premolar),
    }


def find_tooth_column(dataframe):
    def normalize(name):
        return re.sub(r"[^a-z0-9]", "", str(name).lower())

    normalized = {normalize(column): column for column in dataframe.columns}
    for candidate in ["toothnumber", "toothno", "toothnum", "tooth", "toothid"]:
        if candidate in normalized:
            return normalized[candidate]

    for column in dataframe.columns:
        key = normalize(column)
        if "tooth" in key and any(token in key for token in ["number", "num", "no"]):
            return column

    raise KeyError(f"Could not find a tooth-number column. Columns: {list(dataframe.columns)}")


def load_data(annotation_csv, outcome_xlsx, split_csv):
    annotations = pd.read_csv(annotation_csv)
    annotations["case_id"] = annotations["File Name"].map(normalize_case_id)
    annotations = annotations.dropna(subset=["case_id"]).copy()

    outcomes = pd.read_excel(outcome_xlsx)
    tooth_column = find_tooth_column(outcomes)
    outcomes["case_id"] = outcomes["Sequence"].map(normalize_case_id)
    outcomes["label"] = outcomes.apply(prognosis_label, axis=1)
    outcomes["tooth_number"] = pd.to_numeric(outcomes[tooth_column], errors="coerce")
    outcomes = outcomes.dropna(subset=["case_id", "label", "tooth_number"]).copy()
    outcomes["tooth_number"] = outcomes["tooth_number"].astype(int)

    data = annotations.merge(
        outcomes[["case_id", "label", "tooth_number"]],
        on="case_id",
        how="inner",
    )

    data = data.dropna(subset=ANNOTATION_COLUMNS).copy()
    metadata = data["tooth_number"].map(tooth_features).apply(pd.Series)
    features = pd.concat(
        [data[ANNOTATION_COLUMNS].astype(float), metadata],
        axis=1,
    )
    labels = data["label"].astype(int)

    split = pd.read_csv(split_csv)
    split["case_id"] = split["case_id"].map(normalize_case_id)
    split_map = split.set_index("case_id")["split"].to_dict()
    split_names = data["case_id"].map(split_map)

    train_mask = split_names.eq("train")
    val_mask = split_names.eq("val")
    test_mask = split_names.eq("test")
    if not train_mask.any() or not test_mask.any():
        raise ValueError("The split CSV must contain train and test cases.")

    return (
        features.loc[train_mask], labels.loc[train_mask],
        features.loc[val_mask], labels.loc[val_mask],
        features.loc[test_mask], labels.loc[test_mask],
        data.loc[test_mask, "case_id"].to_numpy(),
    )


def evaluate_split(model, X, y):
    """Calculates predictions, probabilities, and metrics for a given dataset split."""
    if len(X) == 0:
        return {"AUC": np.nan, "Bal_Acc": np.nan, "Accuracy": np.nan}

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    auc = roc_auc_score(y, probabilities) if len(np.unique(y)) == 2 else np.nan
    bal_acc = balanced_accuracy_score(y, predictions)
    acc = accuracy_score(y, predictions)

    return {"AUC": auc, "Bal_Acc": bal_acc, "Accuracy": acc}


def get_classifiers(random_state: int):
    """Returns a dictionary of classifier pipelines optimized/suited for binary features."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=random_state, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500, class_weight="balanced", random_state=random_state, n_jobs=-1
        ),
        "Bernoulli Naive Bayes": BernoulliNB(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=random_state
        ),
        "AdaBoost": AdaBoostClassifier(random_state=random_state),
        "Support Vector Machine (RBF)": SVC(
            probability=True, class_weight="balanced", random_state=random_state
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-csv", type=Path, default=Path("ml_ready_cbct_annotation.csv"))
    parser.add_argument("--outcome-xlsx", type=Path, default=Path("Dataset A - Overview.xlsx"))
    parser.add_argument("--split-csv", type=Path, default=Path("checkpoints/prelim_split_seed42.csv"))
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_x, train_y, val_x, val_y, test_x, test_y, test_ids = load_data(
        args.annotation_csv, args.outcome_xlsx, args.split_csv
    )
    print(f"Features ({len(train_x.columns)}): {list(train_x.columns)}")
    print(f"Train / Val / Test rows: {len(train_x)} / {len(val_x)} / {len(test_x)}\n")

    classifiers = get_classifiers(args.random_state)
    results = []
    test_preds_dict = {"case_id": test_ids, "label": test_y.to_numpy()}

    for name, clf in classifiers.items():
        pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), clf)
        pipeline.fit(train_x, train_y)

        val_metrics = evaluate_split(pipeline, val_x, val_y)
        test_metrics = evaluate_split(pipeline, test_x, test_y)

        test_probs = pipeline.predict_proba(test_x)[:, 1]
        test_preds_dict[f"prob_{name.lower().replace(' ', '_')}"] = test_probs

        results.append({
            "Model": name,
            "Val AUC": val_metrics["AUC"],
            "Val Bal Acc": val_metrics["Bal_Acc"],
            "Val Acc": val_metrics["Accuracy"],
            "Test AUC": test_metrics["AUC"],
            "Test Bal Acc": test_metrics["Bal_Acc"],
            "Test Acc": test_metrics["Accuracy"],
        })

    # Format summary table
    summary_df = pd.DataFrame(results).set_index("Model")
    summary_df = summary_df.sort_values(by="Val AUC", ascending=False)

    print("====================================== MODEL COMPARISON ======================================")
    print(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("==============================================================================================\n")

    # Save summary report & predictions
    summary_df.to_csv("classifier_suite_summary.csv")
    pd.DataFrame(test_preds_dict).to_csv("classifier_suite_test_predictions.csv", index=False)
    print("Saved 'classifier_suite_summary.csv' and 'classifier_suite_test_predictions.csv'.")


if __name__ == "__main__":
    main()