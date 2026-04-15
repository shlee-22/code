from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pandas.plotting import parallel_coordinates

from constants import (
    ANGLE_COLUMNS,
    LABEL_COLUMN,
    LABEL_NAME_MAPPING,
    BASE_LABEL_NAMES,
)


# 공통 색상 매핑
COLOR_MAPPING = {
    0: "b",
    1: "g",
    2: "orange",
    3: "black",
    4: "red",
}


def plot_confusion_matrix(cm_percentage: np.ndarray):
    """
    Plot row-normalized confusion matrix heatmap for base labels (0,1,2).
    Returns the created Figure object.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_percentage,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=BASE_LABEL_NAMES,
        yticklabels=BASE_LABEL_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Proportion) for Labels 0, 1, 2")
    plt.tight_layout()
    fig = plt.gcf()
    return fig


def plot_implanted_bar_chart(
    probabilities_by_label: Dict[int, Dict[str, np.ndarray]]
):
    """
    Bar chart comparing prediction distributions for implanted labels (e.g., 3 vs 4).
    Returns the created Figure object.
    """
    base_labels = BASE_LABEL_NAMES
    x = np.arange(len(base_labels))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    # label 3과 4만 있다고 가정 (필요하면 일반화 가능)
    label3_probs = probabilities_by_label.get(3, {}).get("probabilities", None)
    label4_probs = probabilities_by_label.get(4, {}).get("probabilities", None)

    if label3_probs is None or label4_probs is None:
        raise ValueError("Expected implanted labels 3 and 4 in probabilities_by_label.")

    ax.bar(
        x - bar_width / 2,
        label3_probs,
        bar_width,
        label=LABEL_NAME_MAPPING[3],
        color="white",
        edgecolor="black",
    )
    ax.bar(
        x + bar_width / 2,
        label4_probs,
        bar_width,
        label=LABEL_NAME_MAPPING[4],
        color="orange",
        edgecolor="black",
    )

    for i in range(len(base_labels)):
        ax.text(
            x[i] - bar_width / 2,
            label3_probs[i] + 1,
            f"{label3_probs[i]:.2f}%",
            ha="center",
            va="bottom",
        )
        ax.text(
            x[i] + bar_width / 2,
            label4_probs[i] + 1,
            f"{label4_probs[i]:.2f}%",
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Class")
    ax.set_ylabel("Probability (%)")
    ax.set_title("Predictions for Implanted Labels (3 and 4)")
    ax.set_xticks(x)
    ax.set_xticklabels(base_labels)
    ax.set_ylim(-5, 105)
    ax.set_yticks(np.arange(0, 110, 10))
    ax.legend()
    plt.tight_layout()
    return fig


def plot_3d_scatter(df: pd.DataFrame):
    """
    3D scatter plot of hip, knee, ankle angles by label.
    Returns the created Figure object.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for label in sorted(df[LABEL_COLUMN].unique()):
        subset = df[df[LABEL_COLUMN] == label]
        ax.scatter(
            subset["hip_angle"],
            subset["knee_angle"],
            subset["ankle_angle"],
            c=COLOR_MAPPING.get(label, "k"),
            label=LABEL_NAME_MAPPING.get(label, f"Label {label}"),
        )

    ax.set_xlabel("Hip Angle")
    ax.set_ylabel("Knee Angle")
    ax.set_zlabel("Ankle Angle")
    ax.set_title("3D Scatter Plot of Movement Data")
    ax.legend()
    ax.view_init(elev=26, azim=137)
    plt.tight_layout()
    return fig


def plot_radar_chart(df: pd.DataFrame):
    """
    Radar chart of mean Z-scored angles for each label.
    Returns the created Figure object.
    """
    categories = ANGLE_COLUMNS
    N = len(categories)

    overall_mean = df[categories].mean()
    overall_std = df[categories].std()

    df_standardized = df.copy()
    for category in categories:
        df_standardized[category] = (
            df_standardized[category] - overall_mean[category]
        ) / overall_std[category]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories)

    for label in sorted(df[LABEL_COLUMN].unique()):
        label_name = LABEL_NAME_MAPPING.get(label, f"Label {label}")
        values = (
            df_standardized[df_standardized[LABEL_COLUMN] == label][categories]
            .mean()
            .tolist()
        )
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle="solid", label=label_name)
        ax.fill(angles, values, alpha=0.25)

    plt.title("Radar Chart for Labels (Z-scores)", size=14, color="black", y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    return fig


def plot_parallel_coordinates(df: pd.DataFrame):
    """
    Parallel coordinates plot for labels 0, 1, 2.
    Returns the created Figure object.
    """
    df_filtered = df[df[LABEL_COLUMN].isin([0, 1, 2])].copy()
    df_filtered[LABEL_COLUMN] = df_filtered[LABEL_COLUMN].map(LABEL_NAME_MAPPING)

    fig = plt.figure(figsize=(10, 6))
    parallel_coordinates(
        df_filtered,
        class_column=LABEL_COLUMN,
        cols=ANGLE_COLUMNS,
        color=["r", "g", "b"],
    )
    plt.title(
        "Parallel Coordinates Plot for Normal, Incomplete injury, and Complete injury"
    )
    plt.xlabel("Angles")
    plt.ylabel("Values")
    plt.legend(loc="upper right")
    plt.tight_layout()
    return fig
