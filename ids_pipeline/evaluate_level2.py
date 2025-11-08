"""
Đánh giá các mô hình IDS level 2 (theo nhóm) trên tập test và trực quan hóa kết quả.
Mỗi nhóm sẽ đọc pipeline joblib tương ứng trong thư mục artefact, dự đoán trên test.pkl của nhóm, xuất metric và biểu đồ như level 1.

Ví dụ chạy:
python ids_pipeline/evaluate_level2.py \
    --groups dos rare_attack \
    --splits-root dataset/splits/level2 \
    --models-root outputs/level2 \
    --label-column label_encoded \
    --drop-columns label
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate IDS level 2 models on their test sets.")
    parser.add_argument(
        "--groups",
        nargs="*",
        default=["dos", "rare_attack"],
        help="Danh sách nhóm cần đánh giá (mặc định: dos rare_attack).",
    )
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=Path("dataset/splits/level2"),
        help="Thư mục gốc chứa folder từng nhóm (mặc định: dataset/splits/level2).",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path("artifacts_level2"),
        help="Thư mục gốc chứa artefact mô hình từng nhóm (mặc định: artifacts_level2/<group>).",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label_encoded",
        help="Tên cột nhãn chi tiết (mặc định: label_encoded).",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=["label"],
        help="Các cột bỏ trước khi predict (mặc định: label).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/level2_eval"),
        help="Thư mục gốc để lưu reports (mặc định: reports/level2_eval/<group>).",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)8s | %(message)s")


def make_json_safe(value):
    if isinstance(value, dict):
        return {make_json_safe(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, (np.integer, )):
        return int(value)
    if isinstance(value, (np.floating, )):
        return float(value)
    if isinstance(value, (np.ndarray, )):
        return make_json_safe(value.tolist())
    return value


def load_test_dataframe(splits_path: Path, label_column: str, drop_columns: List[str]) -> tuple[pd.DataFrame, pd.Series]:
    test_path = splits_path / "test.pkl"
    if not test_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {test_path}")
    df_test = pd.read_pickle(test_path)
    logging.info("[Eval] Đọc %s: %d dòng, %d cột", test_path, df_test.shape[0], df_test.shape[1])

    column_lookup = {col.lower(): col for col in df_test.columns}
    label_key = label_column.lower()
    if label_key not in column_lookup:
        raise KeyError(f"Không tìm thấy cột nhãn '{label_column}' trong test.pkl")
    label_actual = column_lookup[label_key]

    features = df_test.drop(columns=[label_actual], errors="ignore")
    for col in drop_columns:
        if col:
            features = features.drop(columns=col, errors="ignore")
    labels = df_test[label_actual]
    return features, labels


def plot_metric_bars(class_names: List[str], precision: List[float], recall: List[float], f1: List[float], output_path: Path) -> None:
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1-score")

    ax.set_ylabel("Score")
    ax.set_title("Precision/Recall/F1 per class")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_group(group: str, args: argparse.Namespace) -> None:
    splits_path = (args.splits_root / group).resolve()
    model_path = (args.models_root / group / "ids_pipeline_level2.joblib").resolve()
    output_dir = (args.output_root / group).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=== Evaluate Level 2 group: %s ===", group)
    logging.info("Model path: %s", model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình level 2 cho nhóm '{group}' tại {model_path}. "
            "Hãy chạy ids_pipeline/train_model_level2.py trước hoặc chỉ định --models-root khác."
        )
    pipeline = joblib.load(model_path)

    X_test, y_test = load_test_dataframe(splits_path, args.label_column, args.drop_columns)

    logging.info("Predict trên tập test (%s)...", group)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info("Accuracy: %.6f", accuracy)

    metrics_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

    class_names = [f"class_{i}" for i in range(len(precision))]
    plot_metric_bars(class_names, precision.tolist(), recall.tolist(), f1.tolist(), output_dir / "prf_per_class.png")

    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    cm.ax_.set_title(f"Confusion Matrix - {group.upper()}")
    fig = cm.figure_
    fig.tight_layout()
    handles, labels = cm.ax_.get_legend_handles_labels()
    if labels:
        cm.ax_.legend(handles, labels, title="Predicted", loc="upper right")
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    roc_auc = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
            logging.info("[%s] Macro ROC-AUC: %.6f", group, roc_auc)
        except Exception as exc:
            logging.warning("[%s] Không tính được ROC-AUC: %s", group, exc)

    results_df = pd.DataFrame(
        {
            "class_index": np.arange(len(precision)),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        }
    )
    results_df.to_csv(output_dir / "per_class_metrics.csv", index=False)

    summary = {
        "accuracy": accuracy,
        "roc_auc_macro": roc_auc,
        "classification_report": metrics_report,
    }
    (output_dir / "metrics.json").write_text(json.dumps(make_json_safe(summary), indent=2), encoding="utf-8")

    logging.info("[%s] Đã lưu metrics/biểu đồ tại %s", group, output_dir)


def main() -> None:
    args = parse_args()
    setup_logging()

    for group in args.groups:
        normalized_group = group.strip().lower().replace("-", "_")
        if normalized_group == "rareattack":
            normalized_group = "rare_attack"
        evaluate_group(normalized_group, args)


if __name__ == "__main__":
    main()
