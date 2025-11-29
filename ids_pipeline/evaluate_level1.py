"""
Đánh giá mô hình IDS level 1 (Binary Classification: benign vs attack) trên tập test và trực quan hóa kết quả.

Các bước:
1. Load pipeline đã huấn luyện (joblib).
2. Đọc dữ liệu test đã split.
3. Dự đoán nhãn, tính các metric (accuracy, precision/recall/F1, conf matrix).
4. Vẽ biểu đồ trực quan (Confusion Matrix, ROC curve) và lưu ra file.
5. Xuất thêm bảng metric theo nhãn, biểu đồ so sánh và lưu CSV.

Ví dụ chạy:
python ids_pipeline/evaluate_level1.py \
    --splits-dir dataset/splits/level1 \
    --model-path artifacts_rf/ids_pipeline_rf.joblib \
    --label-column label_binary_encoded \
    --drop-columns label_group label label_group_encoded label_attack_type_encoded \
    --output-dir reports/level1_eval
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
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate IDS level 1 model on test set.")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("dataset/splits/level1"),
        help="Thư mục chứa test.pkl (mặc định: dataset/splits/level1).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts_rf/ids_pipeline_rf.joblib"),
        help="Đường dẫn pipeline đã huấn luyện (mặc định: artifacts_rf/ids_pipeline_rf.joblib).",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label_binary_encoded",
        help="Tên cột nhãn binary (mặc định: label_binary_encoded).",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=["label_group", "label", "label_group_encoded", "label_attack_type_encoded"],
        help="Các cột bỏ trước khi predict (mặc định: label_group, label, label_group_encoded, label_attack_type_encoded).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/level1_eval"),
        help="Thư mục lưu báo cáo và hình ảnh (mặc định: reports/level1_eval).",
    )
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=["benign", "attack"],
        help="Tên hiển thị cho từng class theo thứ tự mã hoá (mặc định: benign attack).",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)8s | %(message)s")


def load_test_dataframe(splits_dir: Path, label_column: str, drop_columns: List[str]) -> tuple[pd.DataFrame, pd.Series]:
    test_path = splits_dir / "test.pkl"
    if not test_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {test_path}. Hãy chạy split_dataset.py trước.")
    df_test = pd.read_pickle(test_path)
    logging.info("Đọc test.pkl: %d dòng, %d cột", df_test.shape[0], df_test.shape[1])

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


def main() -> None:
    args = parse_args()
    setup_logging()

    splits_dir = args.splits_dir.resolve()
    model_path = args.model_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Đang load pipeline từ %s", model_path)
    pipeline = joblib.load(model_path)

    X_test, y_test = load_test_dataframe(splits_dir, args.label_column, args.drop_columns)

    logging.info("Predict trên tập test...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info("Accuracy: %.6f", accuracy)

    metrics_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Precision/Recall/F1 per class for bar chart
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, zero_division=0
    )
    plot_metric_bars(
        args.class_names[: len(precision)],
        precision.tolist(),
        recall.tolist(),
        f1.tolist(),
        output_dir / "prf_per_class.png",
    )

    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    cm.ax_.set_title("Confusion Matrix - Level 1")
    fig = cm.figure_
    fig.tight_layout()
    # Tạo legend thủ công dựa trên nhãn
    handles, labels = cm.ax_.get_legend_handles_labels()
    if labels:
        cm.ax_.legend(handles, labels, title="Predicted", loc="upper right")
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    roc_auc = None
    roc_curve_path = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)
            # Binary classification: dùng class 1 (attack) cho ROC
            if y_proba.shape[1] == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                # Vẽ ROC curve cho binary classification
                roc_display = RocCurveDisplay.from_predictions(
                    y_test, y_proba[:, 1], name="Level 1 (Binary)"
                )
                roc_display.ax_.set_title("ROC Curve - Level 1 (Binary Classification)")
                roc_curve_path = output_dir / "roc_curve.png"
                roc_display.figure_.savefig(roc_curve_path, dpi=150)
                plt.close(roc_display.figure_)
                logging.info("ROC-AUC: %.6f", roc_auc)
                logging.info("Đã lưu ROC curve tại %s", roc_curve_path)
            else:
                # Multi-class: dùng macro average
                roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
                logging.info("Macro ROC-AUC: %.6f", roc_auc)
        except Exception as exc:
            logging.warning("Không tính được ROC-AUC: %s", exc)

    results_df = pd.DataFrame(
        {
            "class_index": np.arange(len(precision)),
            "class_name": args.class_names[: len(precision)],
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

    logging.info("=" * 80)
    logging.info("📊 KẾT QUẢ ĐÁNH GIÁ LEVEL 1 (Binary Classification)")
    logging.info("=" * 80)
    logging.info("Accuracy: %.4f", accuracy)
    if roc_auc is not None:
        logging.info("ROC-AUC: %.4f", roc_auc)
    logging.info("=" * 80)
    logging.info("Đã lưu metrics tại %s", output_dir / "metrics.json")
    logging.info("Đã lưu hình confusion_matrix.png tại %s", output_dir)
    logging.info("Đã lưu hình prf_per_class.png tại %s", output_dir)
    if roc_curve_path:
        logging.info("Đã lưu hình roc_curve.png tại %s", output_dir)
    logging.info("Đã lưu csv per_class_metrics.csv tại %s", output_dir)


if __name__ == "__main__":
    main()
