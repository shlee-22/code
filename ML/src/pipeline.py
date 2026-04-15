
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from data_loader import load_raw_data, split_train_test
from model import build_svm_model, train_model, select_implanted_data
from evaluation import (
    evaluate_classifier,
    compute_confusion_matrix,
    normalize_confusion_matrix,
    fold_report_dict,
    fold_metrics_to_dataframe,
    analyze_implanted_predictions,
)
from visualization import (
    plot_confusion_matrix,
    plot_implanted_bar_chart,
    plot_3d_scatter,
    plot_radar_chart,
    plot_parallel_coordinates,
)
from config import IMPLANTED_LABELS, N_SPLITS
from constants import ANGLE_COLUMNS, LABEL_COLUMN


def run_pipeline(data_file_path=None) -> None:

    # 1) 데이터 로드
    df = load_raw_data(data_file_path)

    # 2) Train/Test split (기존 유지)
    X_train, X_test, y_train, y_test = split_train_test(df)

    # 3) 모델 생성 및 학습
    model = train_model(build_svm_model(), X_train, y_train)

    # 4) 기본 평가 (single split, 기존 유지)
    y_pred, _ = evaluate_classifier(model, X_test, y_test)

    figures = []

    # ==================================================
    # 5) Confusion matrix + fold별 metric (Stratified K-fold)
    # ==================================================
    mask = df[LABEL_COLUMN].isin([0, 1, 2])
    X_all = df.loc[mask, ANGLE_COLUMNS]
    y_all = df.loc[mask, LABEL_COLUMN]

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=42,
    )

    cm_cumulative = np.zeros((3, 3), dtype=int)

    fold_metric_tables = []  # ✅ fold별 raw metric DF를 모을 리스트

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all), start=1):
        X_tr, X_te = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_tr, y_te = y_all.iloc[train_idx], y_all.iloc[test_idx]

        fold_model = train_model(build_svm_model(), X_tr, y_tr)
        y_fold_pred = fold_model.predict(X_te)

        # confusion matrix 누적 (기존)
        cm_cumulative += compute_confusion_matrix(y_te, y_fold_pred)

        # ✅ fold별 precision/recall/f1 raw 값 수집
        report = fold_report_dict(y_te, y_fold_pred)
        fold_df = fold_metrics_to_dataframe(report, fold_idx=fold_idx)
        fold_metric_tables.append(fold_df)

    # (A) 누적 confusion matrix (%)
    cm_percentage = normalize_confusion_matrix(cm_cumulative)
    fig_cm = plot_confusion_matrix(cm_percentage)
    figures.append(fig_cm)

    # (B) ✅ fold별 metrics DataFrame 출력 (Support 제외)
    metrics_df = pd.concat(fold_metric_tables, ignore_index=True)
    print("\n=== K-fold metrics (raw values; no support) ===")
    print(metrics_df.to_string(index=False))

    # 6) Implanted (3,4) 데이터 분석 (기존 유지)
    X_implanted, y_implanted = select_implanted_data(df, IMPLANTED_LABELS)
    if len(X_implanted) > 0:
        implanted_results = analyze_implanted_predictions(model, X_implanted, y_implanted)
        figures.append(plot_implanted_bar_chart(implanted_results))

    # 7~9) 기존 시각화 유지
    figures.append(plot_3d_scatter(df))
    figures.append(plot_radar_chart(df))
    figures.append(plot_parallel_coordinates(df))

    plt.show()