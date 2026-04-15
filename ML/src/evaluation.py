
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator

from constants import BASE_LABEL_NAMES


def evaluate_classifier(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_names=BASE_LABEL_NAMES,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Evaluate classifier on test set and print metrics (single split).
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("\n=== SVM Model Evaluation (Single Split) ===")
    print(f"Accuracy: {accuracy:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=target_names))

    metrics = {"accuracy": accuracy}
    return y_pred, metrics


def compute_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute raw confusion matrix (counts).
    """
    return confusion_matrix(y_true, y_pred)


def normalize_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    """
    Row-normalize confusion matrix to proportions.
    """
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    # divide safely
    return np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)


def analyze_implanted_predictions(
    model: BaseEstimator,
    X_implanted: pd.DataFrame,
    y_implanted: pd.Series,
    base_num_classes: int = 3,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Analyze how implanted labels (3,4) are predicted by a model trained on (0,1,2).
    Returns probabilities (%) per base class.
    """
    y_pred_implanted = model.predict(X_implanted)
    results: Dict[int, Dict[str, np.ndarray]] = {}

    for label in sorted(y_implanted.unique()):
        mask = y_implanted == label
        predictions = y_pred_implanted[mask]
        counts = np.bincount(predictions, minlength=base_num_classes)
        probabilities = counts / counts.sum() * 100.0 if counts.sum() > 0 else np.zeros(base_num_classes)

        results[int(label)] = {"counts": counts, "probabilities": probabilities}

    return results


# =========================
# ✅ 여기부터 "fold별 raw metric DataFrame" 기능
# =========================

def fold_report_dict(
    y_true: pd.Series,
    y_pred: np.ndarray,
    target_names: List[str] = BASE_LABEL_NAMES,
) -> Dict:
    """
    classification_report를 dict 형태로 반환 (output_dict=True).
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )


def fold_metrics_to_dataframe(
    report: Dict,
    fold_idx: int,
    class_names: List[str] = BASE_LABEL_NAMES,
) -> pd.DataFrame:
    """
    한 fold의 report(dict)를 'Support 제외'한 DataFrame으로 변환.
    출력 형태(행): fold / class
    출력 컬럼: precision, recall, f1-score
    """
    rows = []
    for cname in class_names:
        rows.append({
            "fold": fold_idx,
            "class": cname,
            "precision": report[cname]["precision"],
            "recall": report[cname]["recall"],
            "f1-score": report[cname]["f1-score"],
        })

    return pd.DataFrame(rows)