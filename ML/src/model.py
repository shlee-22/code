from typing import Tuple

import pandas as pd
from sklearn.svm import SVC
from sklearn.base import BaseEstimator

from constants import ANGLE_COLUMNS, LABEL_COLUMN


def build_svm_model() -> SVC:
    """
    Construct an SVM classifier with probability estimates enabled.
    """
    model = SVC(probability=True, random_state=42)
    return model


def train_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> BaseEstimator:
    """
    Fit the given model on the training data.
    """
    model.fit(X_train, y_train)
    return model


def select_implanted_data(
    df: pd.DataFrame, implanted_labels
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select rows whose label is in `implanted_labels`.
    """
    mask = df[LABEL_COLUMN].isin(implanted_labels)
    X_new = df.loc[mask, ANGLE_COLUMNS]
    y_new = df.loc[mask, LABEL_COLUMN]
    return X_new, y_new
