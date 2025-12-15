"""
Script huấn luyện Intrusion Detection Model Level 1 với Hybrid CNN+LSTM.

Mô hình Hybrid CNN+LSTM được đề xuất (TOP TREND 2024-2025):
- Input: Sequential features (network traffic flows)
- CNN Blocks: Extract spatial/local features từ network traffic
  * Conv1D(64) -> BatchNorm -> ReLU -> SpatialDropout -> MaxPool1D
  * Conv1D(128) -> BatchNorm -> ReLU -> SpatialDropout -> MaxPool1D
  * Conv1D(256) -> BatchNorm -> ReLU -> SpatialDropout -> MaxPool1D
  * Conv1D(512) -> BatchNorm -> ReLU -> SpatialDropout -> MaxPool1D
- LSTM Layer: Learn temporal patterns và sequence dependencies
  * LSTM(128) với dropout và recurrent dropout
- Dense Layers: Final classification
  * Dense(256) -> Dropout(0.5) -> Dense(128) -> Dropout(0.3) -> Output

Ưu điểm:
✅ CNN: Extract spatial features from network traffic patterns
✅ LSTM: Learn temporal dependencies and behavioral sequences
✅ High accuracy: 92-99% on CICIDS2017 (state-of-the-art)
✅ Trend 2024-2025: Hybrid deep learning for cybersecurity

Pipeline chính:
1. Đảm bảo dữ liệu đã được split (tự chạy scripts/split_dataset.py nếu cần).
2. Đọc các tập train_raw/train_balanced/val/test.
3. Preprocessing cho CNN+LSTM (reshape, normalize).
4. Huấn luyện Hybrid model với advanced regularization.
5. Đánh giá trên validation và holdout/test.
6. Lưu artefact (H5, metrics, metadata).

Ví dụ chạy:
python ids_pipeline/_1d_cnn/train_level1_cnn.py \
    --splits-dir dataset/splits/level1 \
    --train-variant balanced \
    --output-dir artifacts_hybrid \
    --epochs 50 \
    --batch-size 32 \
    --lstm-units 32 \
    --mixed-precision \
    --xla \
    --gpu-memory-limit 8
"""
from __future__ import annotations

# ==================== IMPORTS ====================
import argparse
import json
import logging
import os
from typing import Dict, List, Tuple
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
import joblib

# Ensure reproducibility
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

# ==================== GPU CONFIGURATION ====================
def setup_gpu(gpu_memory_limit=None, gpu_device=None, mixed_precision=False, xla=False):
    """Cấu hình GPU để tăng tốc training"""
    try:
        # Kiểm tra GPU available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            logging.info(f"🔥 Đã tìm thấy {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")

            # Chỉ định GPU device cụ thể nếu được yêu cầu
            if gpu_device is not None:
                gpu_indices = [int(x.strip()) for x in gpu_device.split(',')]
                selected_gpus = [gpus[i] for i in gpu_indices if i < len(gpus)]
                if selected_gpus:
                    tf.config.experimental.set_visible_devices(selected_gpus, 'GPU')
                    logging.info(f"📌 Chỉ sử dụng GPU: {gpu_device}")
                    gpus = selected_gpus

            # Enable memory growth để tránh chiếm hết GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Set memory limit nếu được chỉ định
            if gpu_memory_limit is not None:
                memory_limit_bytes = int(gpu_memory_limit * 1024 * 1024 * 1024)  # Convert GB to bytes
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_bytes)]
                    )
                logging.info(f"📏 Giới hạn GPU memory: {gpu_memory_limit}GB")

            # Log GPU info
            for i, gpu in enumerate(gpus):
                gpu_details = tf.config.experimental.get_device_details(gpu)
                logging.info(f"GPU {i}: {gpu_details}")

        else:
            logging.warning("⚠️  Không tìm thấy GPU. Training sẽ chạy trên CPU (chậm hơn)")

        # Enable mixed precision nếu được yêu cầu
        if mixed_precision:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            logging.info("🚀 Đã enable Mixed Precision training (float16)")

        # Enable XLA nếu được yêu cầu
        if xla:
            tf.config.optimizer.set_jit(True)
            logging.info("⚡ Đã enable XLA optimization")

        # Log TensorFlow version và CUDA info
        logging.info(f"TensorFlow version: {tf.__version__}")
        logging.info(f"CUDA available: {tf.test.is_built_with_cuda()}")
        logging.info(f"cuDNN available: {tf.test.is_built_with_cudnn()}")
        logging.info(f"GPU available: {tf.test.is_gpu_available()}")

    except Exception as e:
        logging.warning(f"Lỗi cấu hình GPU: {e}. Tiếp tục với cấu hình mặc định")


def make_json_safe(value):
    """Chuyển đổi các kiểu numpy/TensorFlow thành kiểu Python native để lưu JSON."""
    if isinstance(value, dict):
        return {make_json_safe(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return make_json_safe(value.tolist())
    if hasattr(value, 'numpy'):  # TensorFlow tensor
        return make_json_safe(value.numpy())
    return value


def parse_args() -> argparse.Namespace:
    """Định nghĩa và parse tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình IDS level 1 với 1D CNN."
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("dataset/splits/level1"),
        help="Thư mục chứa các tập dữ liệu đã chia sẵn (mặc định: dataset/splits/level1).",
    )
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=Path("dataset_clean_cnn.pkl"),
        help="Dataset nguồn dùng để split level 1 nếu chưa có (mặc định: dataset_clean_cnn.pkl).",
    )
    parser.add_argument(
        "--train-variant",
        choices=["raw", "balanced"],
        default="balanced",
        help="Chọn train_raw hay train_balanced để huấn luyện (mặc định: balanced).",
    )
    parser.add_argument(
        "--label-column",
        default="label_binary_encoded",
        help="Tên cột nhãn dùng cho training (mặc định: label_binary_encoded).",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=["label_group", "label", "label_encoded", "label_group_encoded", "label_attack_type_encoded"],
        help="Danh sách cột bỏ qua khi huấn luyện.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed tái lập kết quả.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Nếu muốn dùng một phần train để thử nghiệm (0 < frac ≤ 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts_cnn"),
        help="Thư mục lưu artefact (mô hình, báo cáo, metadata).",
    )
    parser.add_argument(
        "--auto-split",
        action="store_true",
        default=True,
        help="Tự động chạy split_dataset.py level 1 nếu chưa thấy dữ liệu (mặc định bật).",
    )
    parser.add_argument(
        "--no-auto-split",
        dest="auto_split",
        action="store_false",
        help="Tắt tự động split level 1.",
    )
    parser.add_argument(
        "--split-script",
        type=Path,
        default=Path("../scripts/split_dataset.py"),
        help="Đường dẫn script split_dataset.py.",
    )

    # CNN-specific arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Số epochs tối đa để train (mặc định: 20). Với early stopping, thường dừng sớm.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size cho training (mặc định: 128).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate ban đầu (mặc định: 1e-3).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (mặc định: 15).",
    )

    # GPU và Performance arguments
    parser.add_argument(
        "--gpu-memory-limit",
        type=float,
        default=None,
        help="Giới hạn GPU memory (GB). None = sử dụng tất cả.",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Sử dụng mixed precision training (float16) để tăng tốc 2-3x trên GPU.",
    )
    parser.add_argument(
        "--xla",
        action="store_true",
        help="Enable XLA (Accelerated Linear Algebra) để tối ưu performance.",
    )
    parser.add_argument(
        "--gpu-device",
        type=str,
        default=None,
        help="Chỉ định GPU device (ví dụ: '0', '1'). None = sử dụng tất cả GPU.",
    )
    parser.add_argument(
        "--conv-filters",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512],
        help="Số filters cho các conv layers (mặc định: 64 128 256 512).",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=3,
        help="Kernel size cho conv layers (mặc định: 3).",
    )
    parser.add_argument(
        "--dense-units",
        nargs="+",
        type=int,
        default=[256, 128],
        help="Units cho dense layers (mặc định: 256 128).",
    )
    parser.add_argument(
        "--dropout-rates",
        nargs="+",
        type=float,
        default=[0.5, 0.3],
        help="Dropout rates cho dense layers (mặc định: 0.5 0.3).",
    )
    parser.add_argument(
        "--lstm-units",
        type=int,
        default=32,
        help="Số units cho LSTM layer (mặc định: 32). Giảm xuống để tăng tốc training.",
    )
    parser.add_argument(
        "--recurrent-dropout",
        type=float,
        default=0.2,
        help="Recurrent dropout cho LSTM (mặc định: 0.2).",
    )

    return parser.parse_args()


def setup_logging() -> None:
    """Cấu hình logging mức INFO và định dạng thống nhất."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)8s | %(message)s",
    )


def load_split_dataframe(
    path: Path, sample_frac: float | None, random_state: int
) -> pd.DataFrame:
    """Đọc DataFrame từ pickle/CSV và (tuỳ chọn) sample một phần dữ liệu."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {path}")

    logging.info("Đang đọc dữ liệu từ %s", path)
    suffix = path.suffix.lower()

    if suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Định dạng dữ liệu không được hỗ trợ: {suffix}")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Dataset phải là pandas DataFrame sau khi đọc.")

    if sample_frac is not None:
        if not 0 < sample_frac <= 1:
            raise ValueError("--sample-frac phải nằm trong (0, 1].")
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
        logging.info("Sample %.2f => %d rows.", sample_frac, df.shape[0])
    else:
        logging.info("Dataset có %d dòng, %d cột.", df.shape[0], df.shape[1])
    return df


def prepare_cnn_data(
    df: pd.DataFrame,
    feature_columns: List[str],
    label_column: str,
    is_training: bool = True,
    scaler: StandardScaler | None = None
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Chuẩn bị dữ liệu cho CNN 1D:
    - Extract features và labels
    - Standardize features
    - Reshape cho CNN input (samples, timesteps, features)
    """
    logging.info("Chuẩn bị dữ liệu CNN cho %s", "training" if is_training else "inference")

    # Extract features và labels
    X = df[feature_columns].values.astype(np.float32)
    y = df[label_column].values

    # Handle missing values
    if np.isnan(X).any():
        logging.warning("Tìm thấy NaN values, sẽ fill bằng 0")
        X = np.nan_to_num(X, nan=0.0)

    # Standardize features
    if is_training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("Scaler phải được cung cấp cho inference")
        X_scaled = scaler.transform(X)

    # Reshape cho CNN 1D: (samples, timesteps=1, features)
    # Vì network flows là tabular data, chúng ta treat mỗi flow như 1 timestep
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    logging.info("Data shape: %s -> %s", X.shape, X_reshaped.shape)
    logging.info("Label distribution: %s", np.bincount(y) if len(np.unique(y)) < 20 else "Too many classes")

    return X_reshaped, y, scaler


def build_cnn_lstm_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    conv_filters: List[int],
    kernel_size: int,
    lstm_units: int = 32,
    dense_units: List[int] = [128],
    dropout_rates: List[float] = [0.3],
    recurrent_dropout: float = 0.2
) -> keras.Model:
    """
    Xây dựng mô hình Hybrid CNN + LSTM theo kiến trúc đề xuất (TOP TREND 2024-2025).

    Architecture:
    1. CNN Blocks: Extract spatial features từ network traffic patterns
       - Conv1D -> BatchNorm -> ReLU -> SpatialDropout -> MaxPool1D
    2. LSTM Layer: Learn temporal patterns và sequence dependencies
       - LSTM với dropout và recurrent regularization
    3. Dense Layers: Final classification với advanced regularization

    Args:
        input_shape: (timesteps, features)
        num_classes: Số classes đầu ra
        conv_filters: List số filters cho conv blocks
        kernel_size: Kernel size cho conv layers
        lstm_units: Số units cho LSTM layer
        dense_units: List units cho dense layers
        dropout_rates: List dropout rates cho dense layers
        recurrent_dropout: Dropout cho recurrent connections
    """
    logging.info("Xây dựng Hybrid CNN+LSTM model với %d conv blocks + LSTM(%d)",
                len(conv_filters), lstm_units)

    inputs = layers.Input(shape=input_shape)

    # CNN Feature Extraction Blocks
    x = inputs
    for i, filters in enumerate(conv_filters):
        # Conv1D for spatial feature extraction
        x = layers.Conv1D(
            filters,
            kernel_size,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(1e-4),
            name=f'conv_{i+1}'
        )(x)
        x = layers.BatchNormalization(name=f'bn_conv_{i+1}')(x)
        x = layers.ReLU(name=f'relu_conv_{i+1}')(x)

        # Spatial dropout for regularization
        x = layers.SpatialDropout1D(0.1, name=f'spatial_dropout_{i+1}')(x)

        # Max pooling to reduce sequence length while preserving important features
        x = layers.MaxPooling1D(pool_size=2, padding='same', name=f'pool_{i+1}')(x)

    # LSTM for temporal pattern learning
    # LSTM sẽ học dependencies giữa các time steps sau khi CNN extract features
    x = layers.LSTM(
        lstm_units,
        dropout=0.2,
        recurrent_dropout=recurrent_dropout,
        return_sequences=False,  # Return final output only
        kernel_regularizer=keras.regularizers.l2(1e-4),
        recurrent_regularizer=keras.regularizers.l2(1e-4),
        name='lstm_temporal'
    )(x)

    # Additional regularization after LSTM
    x = layers.Dropout(0.3, name='dropout_after_lstm')(x)

    # Dense layers for final classification
    for i, (units, dropout_rate) in enumerate(zip(dense_units, dropout_rates)):
        x = layers.Dense(
            units,
            kernel_regularizer=keras.regularizers.l2(1e-4),
            name=f'dense_{i+1}'
        )(x)
        x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
        x = layers.ReLU(name=f'relu_dense_{i+1}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}')(x)

    # Output layer
    if num_classes == 2:
        # Binary classification
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        loss = 'binary_crossentropy'
        metrics = [
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    else:
        # Multi-class classification
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        loss = 'sparse_categorical_crossentropy'
        metrics = [
            'accuracy',
            keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top_2_accuracy')
        ]

    model = models.Model(inputs=inputs, outputs=outputs, name='ids_cnn_lstm_hybrid')

    # Log model summary
    logging.info("Hybrid CNN+LSTM Architecture:")
    logging.info("CNN blocks: %d | LSTM units: %d | Dense layers: %d",
                len(conv_filters), lstm_units, len(dense_units))
    model.summary(print_fn=lambda x: logging.info(x))

    return model, loss, metrics


def create_callbacks(
    output_dir: Path,
    patience: int,
    model_name: str = "cnn_model"
) -> List[callbacks.Callback]:
    """Tạo callbacks cho training."""
    callbacks_list = [
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),

        # Model checkpoint
        callbacks.ModelCheckpoint(
            filepath=str(output_dir / f"{model_name}_best.h5"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),

        # Learning rate scheduler
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience//2,
            min_lr=1e-6,
            verbose=1
        ),

        # TensorBoard
        callbacks.TensorBoard(
            log_dir=str(output_dir / "tensorboard_logs"),
            histogram_freq=1,
            write_graph=True
        ),

        # CSV Logger
        callbacks.CSVLogger(
            str(output_dir / "training_log.csv"),
            append=False
        )
    ]

    return callbacks_list


def train_and_evaluate(
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    output_dir: Path
) -> Dict:
    """Training và evaluation pipeline."""

    # Prepare data
    logging.info("Preparing training data...")
    X_train, y_train, scaler = prepare_cnn_data(train_df, feature_columns, args.label_column, is_training=True)

    logging.info("Preparing validation data...")
    X_val, y_val, _ = prepare_cnn_data(val_df, feature_columns, args.label_column, is_training=False, scaler=scaler)

    logging.info("Preparing test data...")
    X_test, y_test, _ = prepare_cnn_data(test_df, feature_columns, args.label_column, is_training=False, scaler=scaler)

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    num_classes = len(np.unique(y_train))

    model, loss, metrics = build_cnn_lstm_model(
        input_shape=input_shape,
        num_classes=num_classes,
        conv_filters=args.conv_filters,
        kernel_size=args.kernel_size,
        lstm_units=args.lstm_units,
        dense_units=args.dense_units,
        dropout_rates=args.dropout_rates,
        recurrent_dropout=args.recurrent_dropout
    )

    # Compile model
    optimizer = optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Calculate class weights for imbalanced data
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    logging.info("Class weights: %s", class_weight_dict)

    # Create callbacks
    callbacks_list = create_callbacks(output_dir, args.patience)

    # Train model
    logging.info("Bắt đầu training với %d epochs, batch_size=%d", args.epochs, args.batch_size)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks_list,
        verbose=1
    )

    # Evaluate on test set
    logging.info("Đánh giá trên test set...")
    test_results = model.evaluate(X_test, y_test, verbose=1)

    # Predictions for detailed metrics
    y_pred_proba = model.predict(X_test, verbose=1)
    if num_classes == 2:
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    else:
        y_pred = np.argmax(y_pred_proba, axis=1)

    # Classification report
    target_names = [f'class_{i}' for i in range(num_classes)]
    clf_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Save model and artifacts
    logging.info("Lưu model và artifacts...")

    # Save final model
    model_path = output_dir / "cnn_model_final.h5"
    model.save(model_path)
    logging.info("Model saved to: %s", model_path)

    # Save scaler
    scaler_path = output_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logging.info("Scaler saved to: %s", scaler_path)

    # Save label encoder if needed
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    label_encoder_path = output_dir / "label_encoder.joblib"
    joblib.dump(label_encoder, label_encoder_path)

    # Prepare metadata
    metadata = {
        "model_info": {
            "type": "CNN_LSTM_Hybrid",
            "input_shape": input_shape,
            "num_classes": num_classes,
            "architecture": {
                "conv_filters": args.conv_filters,
                "kernel_size": args.kernel_size,
                "lstm_units": args.lstm_units,
                "recurrent_dropout": args.recurrent_dropout,
                "dense_units": args.dense_units,
                "dropout_rates": args.dropout_rates
            }
        },
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "optimizer": "Adam",
            "loss": loss,
            "metrics": [str(m) for m in metrics]
        },
        "data_info": {
            "feature_columns": feature_columns,
            "label_column": args.label_column,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "class_weights": class_weight_dict
        },
        "performance": {
            "test_loss": float(test_results[0]),
            "test_accuracy": float(test_results[1]),
            "classification_report": make_json_safe(clf_report),
            "confusion_matrix": make_json_safe(conf_matrix)
        },
        "training_history": {
            "epochs_completed": len(history.history['loss']),
            "final_train_loss": float(history.history['loss'][-1]),
            "final_train_accuracy": float(history.history['accuracy'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "final_val_accuracy": float(history.history['val_accuracy'][-1])
        }
    }

    # Save metadata
    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(make_json_safe(metadata), f, indent=2, ensure_ascii=False)
    logging.info("Metadata saved to: %s", metadata_path)

    # Save detailed classification report
    report_path = output_dir / "classification_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(classification_report(y_test, y_pred, target_names=target_names))
        f.write("\n\nCONFUSION MATRIX\n")
        f.write("-" * 30 + "\n")
        f.write(str(conf_matrix))
    logging.info("Classification report saved to: %s", report_path)

    return metadata


def main() -> None:
    """Main function."""
    args = parse_args()

    # Setup GPU trước khi setup logging để log GPU info
    setup_gpu(
        gpu_memory_limit=args.gpu_memory_limit,
        gpu_device=args.gpu_device,
        mixed_precision=args.mixed_precision,
        xla=args.xla
    )

    setup_logging()

    logging.info("🚀 Bắt đầu training IDS Level 1 với CNN 1D")
    logging.info("Arguments: %s", vars(args))

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-split if needed
    if args.auto_split:
        train_path = args.splits_dir / f"train_{args.train_variant}.pkl"
        val_path = args.splits_dir / "val.pkl"
        test_path = args.splits_dir / "test.pkl"

        if not all(p.exists() for p in [train_path, val_path, test_path]):
            logging.info("Không tìm thấy split data, đang chạy auto-split...")

            cmd = [
                sys.executable, str(args.split_script),
                "--level", "1",
                "--output-dir", str(args.splits_dir),
                "--source", str(args.source_dataset),
                "--train-min", "10000",  # Giống Random Forest
                "--train-max", "650000", # GIỚI HẠN BENIGN XUỐNG 650K như Random Forest
                "--random-state", str(args.random_state)
            ]

            try:
                subprocess.run(cmd, check=True)
                logging.info("Auto-split completed successfully")
            except subprocess.CalledProcessError as e:
                logging.error("Auto-split failed: %s", e)
                sys.exit(1)

    # Load datasets
    train_df = load_split_dataframe(
        args.splits_dir / f"train_{args.train_variant}.pkl",
        args.sample_frac,
        args.random_state
    )
    val_df = load_split_dataframe(
        args.splits_dir / "val.pkl",
        None,
        args.random_state
    )
    test_df = load_split_dataframe(
        args.splits_dir / "test.pkl",
        None,
        args.random_state
    )

    # Prepare feature columns
    all_columns = set(train_df.columns)
    drop_columns = set(args.drop_columns)
    feature_columns = list(all_columns - drop_columns - {args.label_column})

    if args.label_column not in train_df.columns:
        raise ValueError(f"Label column '{args.label_column}' not found in dataset")

    logging.info("Features: %d columns", len(feature_columns))
    logging.info("Label: %s", args.label_column)
    logging.info("Training samples: %d", len(train_df))
    logging.info("Validation samples: %d", len(val_df))
    logging.info("Test samples: %d", len(test_df))

    # Train and evaluate
    metadata = train_and_evaluate(args, train_df, val_df, test_df, feature_columns, args.output_dir)

    # Final summary
    logging.info("✅ Training completed!")
    logging.info("📊 Final Test Accuracy: %.4f", metadata['performance']['test_accuracy'])
    logging.info("📁 Artifacts saved to: %s", args.output_dir)
    logging.info("🎯 Best model: %s", args.output_dir / "cnn_model_best.h5")


if __name__ == "__main__":
    main()
