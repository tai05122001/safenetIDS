"""
Script predict tập test với model CNN Level 2 - Attack Types Classification.

Script này:
1. Load model CNN Level 2 đã train từ artifacts_cnn_level2/
2. Load test data từ dataset/splits/level2/test.pkl (chỉ malicious samples)
3. Preprocess và predict attack types (dos, ddos, portscan)
4. Tính metrics (accuracy, precision, recall, F1, confusion matrix)
5. Lưu kết quả và visualization

Ví dụ chạy:
python scripts/predict_test_level2_cnn.py \
    --model-path artifacts_cnn_level2/attack_classifier_cnn_best.h5 \
    --scaler-path artifacts_cnn_level2/scaler.joblib \
    --label-encoder-path artifacts_cnn_level2/label_encoder.joblib \
    --test-data-path dataset/splits/level2/test.pkl \
    --label-column label_attack_type_encoded \
    --drop-columns label_group label label_encoded label_group_encoded label_binary_encoded \
    --output-dir reports/level2_cnn_test_predictions
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict tập test với model CNN Level 1"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts_cnn/cnn_model_best.h5"),
        help="Đường dẫn model CNN (mặc định: artifacts_cnn/cnn_model_best.h5)",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=Path("artifacts_cnn/scaler.joblib"),
        help="Đường dẫn scaler (mặc định: artifacts_cnn/scaler.joblib)",
    )
    parser.add_argument(
        "--label-encoder-path",
        type=Path,
        default=Path("artifacts_cnn/label_encoder.joblib"),
        help="Đường dẫn label encoder (mặc định: artifacts_cnn/label_encoder.joblib)",
    )
    parser.add_argument(
        "--test-data-path",
        type=Path,
        default=Path("dataset/splits/level1/test.pkl"),
        help="Đường dẫn test data (mặc định: dataset/splits/level1/test.pkl)",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label_binary_encoded",
        help="Tên cột nhãn (mặc định: label_binary_encoded)",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=["label_group", "label", "label_encoded", "label_group_encoded", "label_attack_type_encoded"],
        help="Các cột bỏ trước khi predict",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/level1_cnn_test_predictions"),
        help="Thư mục lưu kết quả (mặc định: reports/level1_cnn_test_predictions)",
    )
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=["benign", "attack"],
        help="Tên hiển thị cho từng class (mặc định: benign attack)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size cho prediction (mặc định: 128)",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("artifacts_cnn/training_metadata.json"),
        help="Đường dẫn metadata để load feature columns (mặc định: artifacts_cnn/training_metadata.json)",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Lưu predictions chi tiết vào CSV",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)8s | %(message)s",
    )


def make_json_safe(value):
    """Chuyển đổi numpy types thành Python native types cho JSON."""
    if isinstance(value, dict):
        return {make_json_safe(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if hasattr(value, 'numpy'):  # TensorFlow tensor
        return make_json_safe(value.numpy())
    return value


def load_feature_columns_from_metadata(metadata_path: Path) -> List[str]:
    """Load feature columns từ training metadata để đảm bảo thứ tự đúng."""
    if not metadata_path.exists():
        logging.warning(f"Không tìm thấy metadata tại {metadata_path}, sẽ tự detect features")
        return None

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        feature_columns = metadata.get('data_info', {}).get('feature_columns', [])
        if feature_columns:
            logging.info("Đã load %d feature columns từ metadata", len(feature_columns))
            return feature_columns
    except Exception as e:
        logging.warning(f"Lỗi đọc metadata: {e}")

    return None


def load_test_dataframe(
    test_path: Path, label_column: str, drop_columns: List[str], feature_columns_from_metadata: List[str] = None
) -> tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load test data và extract features, labels."""
    if not test_path.exists():
        raise FileNotFoundError(f"Không tìm thấy test data tại {test_path}")

    logging.info("Đang đọc test data từ %s", test_path)
    df_test = pd.read_pickle(test_path)
    logging.info("Test data: %d dòng, %d cột", df_test.shape[0], df_test.shape[1])

    # Tìm label column (case-insensitive)
    column_lookup = {col.lower(): col for col in df_test.columns}
    label_key = label_column.lower()
    if label_key not in column_lookup:
        raise KeyError(f"Không tìm thấy cột nhãn '{label_column}' trong test data")
    label_actual = column_lookup[label_key]

    # Use feature columns from metadata if available, otherwise auto-detect
    if feature_columns_from_metadata:
        # Verify all required features exist in test data
        missing_features = [col for col in feature_columns_from_metadata if col not in df_test.columns]
        if missing_features:
            raise ValueError(f"Thiếu features trong test data: {missing_features}")

        feature_columns = feature_columns_from_metadata.copy()
        logging.info("✅ Sử dụng feature columns từ metadata (%d features)", len(feature_columns))
    else:
        # Auto-detect features (fallback)
        feature_columns = [col for col in df_test.columns if col != label_actual]
        for col in drop_columns:
            if col in feature_columns:
                feature_columns.remove(col)
        logging.warning("⚠️  Không có metadata, tự detect %d features", len(feature_columns))

    features = df_test[feature_columns]
    labels = df_test[label_actual]

    logging.info("Số features: %d", len(feature_columns))
    logging.info("Số samples: %d", len(features))
    logging.info("Label distribution: %s", labels.value_counts().to_dict())

    return features, labels, feature_columns


def preprocess_for_cnn(
    features: pd.DataFrame,
    scaler: joblib,
    feature_columns: List[str]
) -> np.ndarray:
    """Preprocess features cho CNN model."""
    logging.info("Preprocessing features cho CNN...")

    # Đảm bảo thứ tự cột đúng
    X = features[feature_columns].values.astype(np.float32)

    # Handle missing values
    if np.isnan(X).any():
        logging.warning("Tìm thấy NaN values, sẽ fill bằng 0")
        X = np.nan_to_num(X, nan=0.0)

    # Scale features
    X_scaled = scaler.transform(X)

    # Reshape cho CNN 1D: (samples, timesteps=1, features)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    logging.info("Data shape sau preprocessing: %s", X_reshaped.shape)
    return X_reshaped


def predict_with_cnn(
    model: tf.keras.Model,
    X_test: np.ndarray,
    batch_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """Predict với CNN model."""
    logging.info("Đang predict với batch_size=%d...", batch_size)
    
    # Get probability predictions
    y_pred_proba = model.predict(X_test, batch_size=batch_size, verbose=1)
    
    # Determine if binary or multi-class
    if model.layers[-1].activation.__name__ == 'sigmoid':
        # Binary classification
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    else:
        # Multi-class classification
        y_pred = np.argmax(y_pred_proba, axis=1)
    
    return y_pred, y_pred_proba


def plot_metric_bars(
    class_names: List[str],
    precision: List[float],
    recall: List[float],
    f1: List[float],
    output_path: Path
) -> None:
    """Vẽ bar chart cho precision, recall, F1."""
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width, label="Precision", alpha=0.8)
    ax.bar(x, recall, width, label="Recall", alpha=0.8)
    ax.bar(x + width, f1, width, label="F1-score", alpha=0.8)

    ax.set_ylabel("Score")
    ax.set_title("Precision/Recall/F1 per Class - Level 1 CNN")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logging.info("Đã lưu metric bars tại %s", output_path)


def main() -> None:
    args = parse_args()
    setup_logging()

    # Resolve paths
    model_path = args.model_path.resolve()
    scaler_path = args.scaler_path.resolve()
    label_encoder_path = args.label_encoder_path.resolve()
    test_data_path = args.test_data_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 80)
    logging.info("🚀 BẮT ĐẦU PREDICT TẬP TEST VỚI CNN LEVEL 1")
    logging.info("=" * 80)
    logging.info("Model path: %s", model_path)
    logging.info("Test data path: %s", test_data_path)
    logging.info("Output dir: %s", output_dir)

    # Load model
    logging.info("Đang load model CNN từ %s...", model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model tại {model_path}")
    
    model = tf.keras.models.load_model(str(model_path))
    logging.info("✅ Model loaded successfully")
    model.summary(print_fn=lambda x: logging.info(x))

    # Load scaler
    logging.info("Đang load scaler từ %s...", scaler_path)
    if not scaler_path.exists():
        raise FileNotFoundError(f"Không tìm thấy scaler tại {scaler_path}")
    
    scaler = joblib.load(scaler_path)
    logging.info("✅ Scaler loaded successfully")

    # Load label encoder
    logging.info("Đang load label encoder từ %s...", label_encoder_path)
    if not label_encoder_path.exists():
        raise FileNotFoundError(f"Không tìm thấy label encoder tại {label_encoder_path}")
    
    label_encoder = joblib.load(label_encoder_path)
    logging.info("✅ Label encoder loaded successfully")
    if hasattr(label_encoder, 'classes_'):
        logging.info("Classes: %s", list(label_encoder.classes_))

    # Load feature columns from metadata để đảm bảo thứ tự đúng
    feature_columns_from_metadata = load_feature_columns_from_metadata(args.metadata_path)

    # Load test data
    features_test, labels_test, feature_columns = load_test_dataframe(
        test_data_path, args.label_column, args.drop_columns, feature_columns_from_metadata
    )

    # Preprocess
    X_test = preprocess_for_cnn(features_test, scaler, feature_columns)

    # Predict
    y_pred, y_pred_proba = predict_with_cnn(model, X_test, args.batch_size)

    # Convert labels to numpy if needed
    y_test = labels_test.values if isinstance(labels_test, pd.Series) else labels_test

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    logging.info("=" * 80)
    logging.info("📊 KẾT QUẢ PREDICTION")
    logging.info("=" * 80)
    logging.info("Accuracy: %.6f (%.2f%%)", accuracy, accuracy * 100)

    # Classification report
    target_names = args.class_names[:len(np.unique(y_test))]
    metrics_report = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    
    # Print classification report
    logging.info("\nClassification Report:")
    logging.info(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # Precision/Recall/F1 per class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, zero_division=0
    )

    # Plot metric bars
    plot_metric_bars(
        target_names,
        precision.tolist(),
        recall.tolist(),
        f1.tolist(),
        output_dir / "prf_per_class.png",
    )

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    cm_display.plot(cmap='Blues', values_format='d')
    cm_display.ax_.set_title("Confusion Matrix - Level 1 CNN")
    fig = cm_display.figure_
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    logging.info("Đã lưu confusion matrix tại %s", output_dir / "confusion_matrix.png")

    # ROC Curve (chỉ cho binary classification)
    roc_auc = None
    roc_curve_path = None
    if len(np.unique(y_test)) == 2:
        try:
            # Binary classification: use probability of class 1
            if y_pred_proba.shape[1] == 1:
                # Sigmoid output (binary)
                y_proba_for_roc = y_pred_proba.flatten()
            else:
                # Softmax output (multi-class but only 2 classes)
                y_proba_for_roc = y_pred_proba[:, 1]
            
            roc_auc = roc_auc_score(y_test, y_proba_for_roc)
            roc_display = RocCurveDisplay.from_predictions(
                y_test, y_proba_for_roc, name="Level 1 CNN (Binary)"
            )
            roc_display.ax_.set_title("ROC Curve - Level 1 CNN (Binary Classification)")
            roc_display.ax_.plot([0, 1], [0, 1], "k--", label="Random")
            roc_display.ax_.legend()
            roc_curve_path = output_dir / "roc_curve.png"
            roc_display.figure_.savefig(roc_curve_path, dpi=150)
            plt.close(roc_display.figure_)
            logging.info("ROC-AUC: %.6f", roc_auc)
            logging.info("Đã lưu ROC curve tại %s", roc_curve_path)
        except Exception as exc:
            logging.warning("Không thể tính ROC-AUC: %s", exc)

    # Save per-class metrics to CSV
    results_df = pd.DataFrame({
        "class_index": np.arange(len(precision)),
        "class_name": target_names[:len(precision)],
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": support,
    })
    results_df.to_csv(output_dir / "per_class_metrics.csv", index=False)
    logging.info("Đã lưu per-class metrics tại %s", output_dir / "per_class_metrics.csv")

    # Save predictions to CSV if requested
    if args.save_predictions:
        predictions_df = pd.DataFrame({
            "true_label": y_test,
            "predicted_label": y_pred,
            "confidence": np.max(y_pred_proba, axis=1) if len(y_pred_proba.shape) > 1 else y_pred_proba.flatten(),
        })
        
        # Add probability columns if binary
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] <= 2:
            for i, class_name in enumerate(target_names[:y_pred_proba.shape[1]]):
                predictions_df[f"prob_{class_name}"] = y_pred_proba[:, i]
        
        predictions_df.to_csv(output_dir / "predictions.csv", index=False)
        logging.info("Đã lưu predictions chi tiết tại %s", output_dir / "predictions.csv")

    # Save summary metrics to JSON
    summary = {
        "model_path": str(model_path),
        "test_data_path": str(test_data_path),
        "test_samples": len(y_test),
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "per_class_metrics": {
            "class_names": target_names[:len(precision)],
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1_score": f1.tolist(),
            "support": support.tolist(),
        },
        "confusion_matrix": make_json_safe(cm.tolist()),
        "classification_report": make_json_safe(metrics_report),
    }

    summary_path = output_dir / "prediction_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(make_json_safe(summary), f, indent=2, ensure_ascii=False)
    logging.info("Đã lưu summary tại %s", summary_path)

    # Final summary
    logging.info("=" * 80)
    logging.info("✅ HOÀN THÀNH PREDICTION")
    logging.info("=" * 80)
    logging.info("📊 Accuracy: %.4f%%", accuracy * 100)
    if roc_auc is not None:
        logging.info("📈 ROC-AUC: %.4f", roc_auc)
    logging.info("📁 Kết quả đã lưu tại: %s", output_dir)
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

