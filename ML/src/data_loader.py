from typing import Tuple, Optional, Union
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_FILE_PATH, TRAIN_LABELS, TEST_SIZE, RANDOM_STATE
from constants import ANGLE_COLUMNS, LABEL_COLUMN


def load_raw_data(data_file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load the movement data from the Excel file and keep only relevant columns.
    """
    if data_file_path is None:
        path = DATA_FILE_PATH
    else:
        path = Path(data_file_path)

    df = pd.read_excel(path)
    df = df[ANGLE_COLUMNS + [LABEL_COLUMN]]
    return df


def split_train_test(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Filter labels to TRAIN_LABELS and perform train/test split.
    """
    mask = df[LABEL_COLUMN].isin(TRAIN_LABELS)
    df_train = df[mask].copy()

    X = df_train[ANGLE_COLUMNS]
    y = df_train[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
