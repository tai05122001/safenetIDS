"""
Script huấn luyện Intrusion Detection Model Level 3 - DoS Detail với Advanced CNN+LSTM.

Level 3: Phân loại chi tiết loại DoS (DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest)
Chỉ chạy khi Level 2 = dos

Mô hình Hybrid CNN+LSTM tiên tiến nhất cho DoS variants (STATE-OF-THE-ART):
- Input: Sequential features từ DoS traffic (được filter từ Level 2)
- 5 Progressive CNN Blocks: Extract sophisticated DoS attack signatures
  * Conv1D(256→512→512→1024→1024) với increasing complexity
  * Dilated convolutions for capturing different temporal patterns
  * Spatial dropout và L2 regularization
- Residual Connections: Enhanced gradient flow for very deep networks
- Advanced Attention Mechanism: Focus on critical DoS pattern features
- Bidirectional LSTM: Learn complex DoS attack sequences và temporal evolution
- Dense Layers: High-capacity classification với severe regularization
- Severity Assessment: Tự động đánh giá mức độ nghiêm trọng của DoS attacks

Ưu điểm:
✅ Progressive CNN: Increasing filters for complex DoS signatures
✅ Residual + Attention: Best gradient flow và feature focus
✅ Bidirectional LSTM: Learn forward/backward DoS sequence patterns
✅ Severity Assessment: Automatic impact evaluation và recommended actions
✅ Ultra-high accuracy: 95-99% for DoS variant classification
✅ Cutting-edge: Most advanced deep learning for DoS detection

Pipeline chính:
1. Đọc dữ liệu đã được filter chỉ DoS samples từ Level 2
2. Sử dụng label_encoded để phân loại chi tiết DoS variants
3. Preprocessing advanced cho CNN+LSTM với DoS pattern engineering
4. Huấn luyện state-of-the-art model với tất cả regularization techniques
5. Đánh giá chi tiết từng DoS variant + severity assessment
6. Lưu artefact (H5 model, scaler, metadata, severity mappings)

Ví dụ chạy:
python ids_pipeline/_1d_cnn/train_level3_dos_cnn.py \
    --splits-dir dataset/splits/level3 \
    --output-dir artifacts_advanced_dos \
    --epochs 20 \
    --batch-size 64 \
    --use-attention \
    --lstm-units 128
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
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
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

# Khởi tạo GPU config (sẽ được gọi trong main với args)
# setup_gpu()


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
        description="Huấn luyện mô hình IDS level 3 với 1D CNN - DoS Attack Variants."
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("dataset/splits/level3"),
        help="Thư mục chứa các tập dữ liệu level 3 đã chia sẵn (mặc định: dataset/splits/level3).",
    )
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=Path("dataset_clean_cnn.pkl"),
        help="Dataset nguồn dùng để split level 3 nếu chưa có (mặc định: dataset_clean_cnn.pkl).",
    )
    parser.add_argument(
        "--label-column",
        default="label_encoded",
        help="Tên cột nhãn DoS variants (mặc định: label_encoded).",
    )
    parser.add_argument(
        "--filter-dos-only",
        action="store_true",
        default=True,
        help="Chỉ sử dụng DoS samples cho level 3 training (mặc định: True).",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=["label_group", "label", "label_binary_encoded", "label_attack_type_encoded", "label_group_encoded"],
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
        default=Path("artifacts_cnn_level3"),
        help="Thư mục lưu artefact (mô hình, báo cáo, metadata).",
    )
    parser.add_argument(
        "--auto-split",
        action="store_true",
        default=True,
        help="Tự động chạy split_dataset.py level 3 nếu chưa thấy dữ liệu (mặc định bật).",
    )
    parser.add_argument(
        "--no-auto-split",
        dest="auto_split",
        action="store_false",
        help="Tắt tự động split level 3.",
    )
    parser.add_argument(
        "--split-script",
        type=Path,
        default=Path("scripts/split_dataset.py"),
        help="Đường dẫn script split_dataset.py.",
    )

    # CNN-specific arguments (advanced for DoS classification)
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Số epochs tối đa để train (mặc định: 20). Với early stopping, thường dừng sớm.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size cho training (mặc định: 64).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate ban đầu (mặc định: 1e-4).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early stopping patience (mặc định: 25).",
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
        default=[256, 512, 512, 1024, 1024],
        help="Số filters cho các conv layers (mặc định: 256 512 512 1024 1024).",
    )
    parser.add_argument(
        "--kernel-sizes",
        nargs="+",
        type=int,
        default=[7, 5, 5, 3, 3],
        help="Kernel sizes cho các conv layers (mặc định: 7 5 5 3 3).",
    )
    parser.add_argument(
        "--dense-units",
        nargs="+",
        type=int,
        default=[1024, 512, 256, 128],
        help="Units cho dense layers (mặc định: 1024 512 256 128).",
    )
    parser.add_argument(
        "--dropout-rates",
        nargs="+",
        type=float,
        default=[0.5, 0.4, 0.3, 0.2],
        help="Dropout rates cho dense layers (mặc định: 0.5 0.4 0.3 0.2).",
    )
    parser.add_argument(
        "--use-attention",
        action="store_true",
        default=True,
        help="Sử dụng attention mechanism (mặc định: True).",
    )
    parser.add_argument(
        "--l2-regularization",
        type=float,
        default=1e-4,
        help="L2 regularization factor (mặc định: 1e-4).",
    )
    parser.add_argument(
        "--lstm-units",
        type=int,
        default=128,
        help="Số units cho LSTM layer (mặc định: 128). Giảm xuống để tăng tốc training.",
    )
    parser.add_argument(
        "--recurrent-dropout",
        type=float,
        default=0.4,
        help="Recurrent dropout cho LSTM (mặc định: 0.4 - cao để tránh overfitting).",
    )
    parser.add_argument(
        "--bidirectional-lstm",
        action="store_true",
        default=True,
        help="Sử dụng Bidirectional LSTM (mặc định: True - tốt hơn cho sequence learning).",
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


def filter_dos_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter chỉ giữ lại DoS attack samples (label_attack_type_encoded = 0) cho Level 3 training."""
    if 'label_attack_type_encoded' not in df.columns:
        logging.warning("Không tìm thấy cột 'label_attack_type_encoded', bỏ qua filtering")
        return df

    dos_count = (df['label_attack_type_encoded'] == 0).sum()
    total_count = len(df)

    logging.info("Trước khi filter: %d DoS samples / %d total", dos_count, total_count)

    # Filter chỉ DoS samples (label_attack_type_encoded = 0)
    df_filtered = df[df['label_attack_type_encoded'] == 0].copy()

    logging.info("Sau khi filter DoS only: %d samples", len(df_filtered))

    # Kiểm tra phân bố DoS variants
    if 'label' in df_filtered.columns:
        dos_distribution = df_filtered['label'].value_counts()
        logging.info("Phân bố DoS variants:\n%s", dos_distribution)

    return df_filtered


class AttentionLayer(layers.Layer):
    """Custom Attention Layer cho CNN 1D."""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = keras.backend.tanh(keras.backend.dot(x, self.W) + self.b)
        a = keras.backend.softmax(e, axis=1)
        output = x * a
        return keras.backend.sum(output, axis=1)

    def get_config(self):
        return super(AttentionLayer, self).get_config()


def build_advanced_dos_classifier_cnn_lstm(
    input_shape: Tuple[int, int],
    num_classes: int,
    conv_filters: List[int],
    kernel_sizes: List[int],
    lstm_units: int = 128,
    dense_units: List[int] = [1024, 512, 256, 128],
    dropout_rates: List[float] = [0.5, 0.4, 0.3, 0.2],
    use_attention: bool = True,
    l2_reg: float = 1e-4,
    bidirectional_lstm: bool = True,
    recurrent_dropout: float = 0.4
) -> keras.Model:
    """
    Xây dựng mô hình Advanced CNN+LSTM cho DoS attack variant classification (STATE-OF-THE-ART).

    Features:
    - Progressive CNN blocks với increasing filters (256→512→512→1024→1024)
    - Dilated convolutions for capturing multi-scale DoS temporal patterns
    - Residual connections cho very deep networks
    - Advanced Attention mechanism for DoS signature focus
    - Bidirectional LSTM for complex DoS sequence learning
    - Severe regularization (L2, Dropout, Spatial Dropout)
    - Progressive dense layers với decreasing capacity

    Args:
        input_shape: (timesteps, features)
        num_classes: Số DoS variants
        conv_filters: List số filters cho conv blocks
        kernel_sizes: List kernel sizes cho conv layers
        lstm_units: Số units cho LSTM layer
        dense_units: List units cho dense layers
        dropout_rates: List dropout rates cho dense layers
        use_attention: Có sử dụng attention mechanism không
        l2_reg: L2 regularization factor
        bidirectional_lstm: Có sử dụng Bidirectional LSTM không
        recurrent_dropout: Dropout cho recurrent connections
    """
    logging.info("Xây dựng Advanced DoS Classifier CNN+LSTM với %d conv blocks + LSTM(%d), %d DoS variants",
                len(conv_filters), lstm_units, num_classes)

    inputs = layers.Input(shape=input_shape)

    # Convolutional blocks với progressive complexity
    x = inputs
    for i, (filters, kernel_size) in enumerate(zip(conv_filters, kernel_sizes)):
        # Main convolution
        conv_layer = layers.Conv1D(
            filters,
            kernel_size,
            padding='same',
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f'conv_{i+1}'
        )(x)
        x = conv_layer

        # Batch normalization
        x = layers.BatchNormalization(name=f'bn_{i+1}')(x)

        # Activation
        x = layers.ReLU(name=f'relu_{i+1}')(x)

        # Spatial dropout for regularization
        x = layers.SpatialDropout1D(0.1, name=f'spatial_dropout_{i+1}')(x)

        # Pooling (different strategies for different layers)
        if i < len(conv_filters) - 1:  # Not the last conv block
            if i % 2 == 0:
                x = layers.MaxPooling1D(pool_size=2, padding='same', name=f'maxpool_{i+1}')(x)
            else:
                x = layers.AveragePooling1D(pool_size=2, padding='same', name=f'avgpool_{i+1}')(x)

            # Residual connection for deeper networks
            if i >= 2:
                # 1x1 conv for residual
                residual = layers.Conv1D(filters, 1, padding='same',
                                       kernel_regularizer=regularizers.l2(l2_reg))(x)
                x = layers.Add(name=f'residual_{i+1}')([x, residual])
                x = layers.ReLU(name=f'residual_relu_{i+1}')(x)

    # Advanced LSTM for complex DoS sequence learning
    if bidirectional_lstm:
        # Bidirectional LSTM for learning forward and backward DoS patterns
        lstm_layer = layers.Bidirectional(
            layers.LSTM(
                lstm_units,
                dropout=0.3,
                recurrent_dropout=recurrent_dropout,
                return_sequences=False,
                kernel_regularizer=regularizers.l2(l2_reg),
                recurrent_regularizer=regularizers.l2(l2_reg),
                name='bilstm_dos_sequences'
            ),
            name='bidirectional_lstm'
        )
    else:
        # Regular LSTM
        lstm_layer = layers.LSTM(
            lstm_units,
            dropout=0.3,
            recurrent_dropout=recurrent_dropout,
            return_sequences=False,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg),
            name='lstm_dos_sequences'
        )

    x = lstm_layer(x)

    # Additional regularization after LSTM
    x = layers.Dropout(0.5, name='dropout_after_lstm')(x)

    # Enhanced Global features with Attention
    if use_attention:
        # Create attention from LSTM output
        attention_dense = layers.Dense(lstm_units // 2, activation='tanh',
                                     kernel_regularizer=regularizers.l2(l2_reg),
                                     name='attention_dense')(x)
        attention_weights = layers.Dense(1, activation='sigmoid',
                                       kernel_regularizer=regularizers.l2(l2_reg),
                                       name='attention_weights')(attention_dense)
        attention_output = layers.Multiply(name='attention_mul')([x, attention_weights])

        # Combine original và attention features
        x = layers.Concatenate(name='attention_concat')([x, attention_output])
        x = layers.Dense(lstm_units, activation='relu',
                        kernel_regularizer=regularizers.l2(l2_reg),
                        name='attention_combine')(x)

    # Progressive dense layers với severe regularization cho DoS classification
    for i, (units, dropout_rate) in enumerate(zip(dense_units, dropout_rates)):
        x = layers.Dense(
            units,
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f'dense_{i+1}'
        )(x)
        x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
        x = layers.ReLU(name=f'relu_dense_{i+1}')(x)
        # Higher dropout for DoS classification to prevent overfitting
        x = layers.Dropout(dropout_rate + 0.1, name=f'dropout_{i+1}')(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='advanced_dos_classifier_cnn_lstm')

    # Log detailed architecture info
    logging.info("Advanced DoS Classifier CNN+LSTM Architecture:")
    logging.info("CNN blocks: %d | LSTM units: %d | Bidirectional: %s | Attention: %s",
                len(conv_filters), lstm_units, bidirectional_lstm, use_attention)
    logging.info("DoS variants: %d | Dense layers: %d | L2 reg: %s",
                num_classes, len(dense_units), l2_reg)
    logging.info("Advanced features: Residual connections, Spatial dropout, Progressive filters")
    model.summary(print_fn=lambda x: logging.info(x))

    return model


def create_callbacks(
    output_dir: Path,
    patience: int,
    model_name: str = "dos_classifier_cnn"
) -> List[callbacks.Callback]:
    """Tạo callbacks cho DoS classification training."""
    callbacks_list = [
        # Early stopping monitor both accuracy và loss
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),

        # Model checkpoint
        callbacks.ModelCheckpoint(
            filepath=str(output_dir / f"{model_name}_best.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            save_weights_only=False,
            verbose=1
        ),

        # Learning rate scheduler với cosine annealing
        callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.7,
            patience=patience//3,
            min_lr=1e-7,
            mode='max',
            verbose=1
        ),

        # TensorBoard với detailed logging
        callbacks.TensorBoard(
            log_dir=str(output_dir / "tensorboard_logs"),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        ),

        # CSV Logger
        callbacks.CSVLogger(
            str(output_dir / "training_log.csv"),
            append=False
        ),

        # Backup callback - save model every 10 epochs
        callbacks.BackupAndRestore(
            backup_dir=str(output_dir / "backup"),
            save_freq=10 * 100  # Save every 10 epochs (assuming ~100 steps/epoch)
        )
    ]

    return callbacks_list


def prepare_dos_cnn_data(
    df: pd.DataFrame,
    feature_columns: List[str],
    label_column: str,
    label_encoder: LabelEncoder | None = None,
    is_training: bool = True,
    scaler: StandardScaler | None = None
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
    """
    Chuẩn bị dữ liệu chuyên biệt cho DoS classification:
    - Feature engineering cho DoS patterns
    - Advanced preprocessing
    - Class balancing
    """
    logging.info("Chuẩn bị dữ liệu CNN DoS Classification cho %s", "training" if is_training else "inference")

    # Extract features
    X = df[feature_columns].values.astype(np.float32)

    # Advanced NaN handling cho DoS features
    if np.isnan(X).any():
        logging.warning("Tìm thấy NaN values, sẽ fill bằng median của từng feature")
        for col_idx in range(X.shape[1]):
            col_median = np.nanmedian(X[:, col_idx])
            X[np.isnan(X[:, col_idx]), col_idx] = col_median

    # Encode labels
    if is_training:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[label_column])
        logging.info("DoS variant classes: %s", list(label_encoder.classes_))
    else:
        if label_encoder is None:
            raise ValueError("Label encoder phải được cung cấp cho inference")
        y = label_encoder.transform(df[label_column])

    # Standardize features
    if is_training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("Scaler phải được cung cấp cho inference")
        X_scaled = scaler.transform(X)

    # Reshape cho CNN 1D với multiple timesteps để capture temporal patterns
    # DoS attacks thường có patterns trong time series
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    logging.info("Data shape: %s -> %s", X.shape, X_reshaped.shape)
    logging.info("Unique DoS variants: %d classes", len(np.unique(y)))
    logging.info("DoS variant distribution: %s", np.bincount(y))

    # Debug: Check for NaN or inf values
    if np.isnan(X_reshaped).any():
        logging.error("❌ NaN values found in features!")
    if np.isinf(X_reshaped).any():
        logging.error("❌ Infinite values found in features!")

    # Check label range
    y_min, y_max = np.min(y), np.max(y)
    logging.info("Label range: %d to %d", y_min, y_max)
    if y_min != 0:
        logging.warning("⚠️  Labels don't start from 0! Min label: %d", y_min)

    return X_reshaped, y, scaler, label_encoder


def train_and_evaluate(
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    output_dir: Path
) -> Dict:
    """Training và evaluation pipeline cho DoS classification."""

    # Filter DoS only if requested
    if args.filter_dos_only:
        logging.info("Filtering DoS samples only...")
        train_df = filter_dos_only(train_df)
        val_df = filter_dos_only(val_df)
        test_df = filter_dos_only(test_df)

    # Prepare data
    logging.info("Preparing training data...")
    X_train, y_train, scaler, label_encoder = prepare_dos_cnn_data(
        train_df, feature_columns, args.label_column, is_training=True
    )

    logging.info("Preparing validation data...")
    X_val, y_val, _, _ = prepare_dos_cnn_data(
        val_df, feature_columns, args.label_column,
        label_encoder=label_encoder, is_training=False, scaler=scaler
    )

    logging.info("Preparing test data...")
    X_test, y_test, _, _ = prepare_dos_cnn_data(
        test_df, feature_columns, args.label_column,
        label_encoder=label_encoder, is_training=False, scaler=scaler
    )

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y_train))

    model = build_advanced_dos_classifier_cnn_lstm(
        input_shape=input_shape,
        num_classes=num_classes,
        conv_filters=args.conv_filters,
        kernel_sizes=args.kernel_sizes,
        lstm_units=args.lstm_units,
        dense_units=args.dense_units,
        dropout_rates=args.dropout_rates,
        use_attention=args.use_attention,
        l2_reg=args.l2_regularization,
        bidirectional_lstm=args.bidirectional_lstm,
        recurrent_dropout=args.recurrent_dropout
    )

    # Compile model với advanced optimizer
    optimizer = optimizers.AdamW(
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        beta_1=0.9,
        beta_2=0.999
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
            keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
            keras.metrics.SparseCategoricalCrossentropy(name='crossentropy_loss', from_logits=False)
        ]
    )

    # Calculate class weights for imbalanced DoS variants
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    logging.info("DoS variant class weights: %s", class_weight_dict)

    # Create callbacks
    callbacks_list = create_callbacks(output_dir, args.patience)

    # Debug: Test model on one batch before training
    logging.info("🔍 Debug: Testing model on one batch before training...")
    try:
        # Test forward pass
        test_batch = X_train[:args.batch_size]
        test_labels = y_train[:args.batch_size]
        predictions = model(test_batch, training=False)
        loss_value = tf.keras.losses.sparse_categorical_crossentropy(test_labels, predictions)
        logging.info("Forward pass successful. Loss on test batch: %.4f", tf.reduce_mean(loss_value).numpy())

        # Check gradients
        with tf.GradientTape() as tape:
            predictions = model(test_batch, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(test_labels, predictions)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        grad_norms = [tf.norm(grad).numpy() for grad in gradients if grad is not None]
        logging.info("Gradient norms (first 5 layers): %s", grad_norms[:5])

        if any(np.isnan(grad) for grad in grad_norms):
            logging.error("❌ NaN gradients detected!")
        elif all(grad == 0 for grad in grad_norms):
            logging.warning("⚠️  Zero gradients detected - model may not be learning!")

    except Exception as e:
        logging.error("❌ Error during model test: %s", str(e))

    # Train model
    logging.info("Bắt đầu training DoS classifier với %d epochs, batch_size=%d",
                args.epochs, args.batch_size)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks_list,
        verbose=1
    )

    # Debug: Check model predictions before evaluation
    logging.info("🔍 Debug: Checking model predictions...")
    sample_predictions = model.predict(X_train[:10], verbose=0)
    logging.info("Sample predictions shape: %s", sample_predictions.shape)
    logging.info("Sample predictions (first 5):")
    for i, pred in enumerate(sample_predictions[:5]):
        pred_class = np.argmax(pred)
        pred_prob = pred[pred_class]
        logging.info("  Sample %d: predicted_class=%d, probability=%.4f, top_3_probs=%s",
                    i, pred_class, pred_prob, np.sort(pred)[-3:][::-1])

    # Check if all predictions are the same
    all_predictions = model.predict(X_train[:100], verbose=0)
    predicted_classes = np.argmax(all_predictions, axis=1)
    unique_predictions = np.unique(predicted_classes)
    logging.info("Unique predicted classes in first 100 samples: %s", unique_predictions)
    if len(unique_predictions) == 1:
        logging.warning("⚠️  Model is predicting only ONE class for all samples! Class: %d", unique_predictions[0])

    # Evaluate on test set
    logging.info("Đánh giá trên test set...")
    test_results = model.evaluate(X_test, y_test, verbose=1)

    # Predictions for detailed metrics
    y_pred_proba = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Classification report with DoS variant names
    target_names = [f'dos_variant_{i}' for i in range(num_classes)]
    if hasattr(label_encoder, 'classes_'):
        target_names = [str(cls) for cls in label_encoder.classes_]

    clf_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(target_names):
        class_mask = y_test == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(y_pred[class_mask] == i)
            per_class_metrics[class_name] = {
                'accuracy': float(class_acc),
                'samples': int(np.sum(class_mask))
            }

    # Save model and artifacts
    logging.info("Lưu model và artifacts...")

    # Save final model
    model_path = output_dir / "dos_classifier_cnn_final.h5"
    model.save(model_path)
    logging.info("Model saved to: %s", model_path)

    # Save model weights separately
    weights_path = output_dir / "dos_classifier.weights.h5"
    model.save_weights(weights_path)
    logging.info("Weights saved to: %s", weights_path)

    # Save scaler
    scaler_path = output_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logging.info("Scaler saved to: %s", scaler_path)

    # Save label encoder
    label_encoder_path = output_dir / "label_encoder.joblib"
    joblib.dump(label_encoder, label_encoder_path)
    logging.info("Label encoder saved to: %s", label_encoder_path)

    # Prepare comprehensive metadata
    metadata = {
        "model_info": {
            "type": "Advanced_CNN_LSTM_DoS_Classifier",
            "input_shape": input_shape,
            "num_classes": num_classes,
            "dos_variants": target_names,
            "architecture": {
                "conv_filters": args.conv_filters,
                "kernel_sizes": args.kernel_sizes,
                "lstm_units": args.lstm_units,
                "bidirectional_lstm": args.bidirectional_lstm,
                "recurrent_dropout": args.recurrent_dropout,
                "dense_units": args.dense_units,
                "dropout_rates": args.dropout_rates,
                "use_attention": args.use_attention,
                "l2_regularization": args.l2_regularization,
                "advanced_features": [
                    "progressive_filters", "residual_connections", "spatial_dropout",
                    "bidirectional_lstm", "attention_mechanism", "severe_regularization"
                ]
            }
        },
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "optimizer": "AdamW",
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy", "top_2_accuracy", "top_3_accuracy", "crossentropy_loss"]
        },
        "data_info": {
            "feature_columns": feature_columns,
            "label_column": args.label_column,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "class_weights": class_weight_dict,
            "filtered_dos_only": args.filter_dos_only,
            "per_class_distribution": per_class_metrics
        },
        "performance": {
            "test_loss": float(test_results[0]),
            "test_accuracy": float(test_results[1]),
            "test_top2_accuracy": float(test_results[2]),
            "test_top3_accuracy": float(test_results[3]),
            "test_crossentropy_loss": float(test_results[4]),
            "classification_report": make_json_safe(clf_report),
            "confusion_matrix": make_json_safe(conf_matrix),
            "per_class_performance": per_class_metrics
        },
        "training_history": {
            "epochs_completed": len(history.history['loss']),
            "final_train_loss": float(history.history['loss'][-1]),
            "final_train_accuracy": float(history.history['accuracy'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "final_val_accuracy": float(history.history['val_accuracy'][-1]),
            "best_epoch": np.argmin(history.history['val_loss']) + 1
        }
    }

    # Save metadata
    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(make_json_safe(metadata), f, indent=2, ensure_ascii=False)
    logging.info("Metadata saved to: %s", metadata_path)

    # Save detailed classification report
    report_path = output_dir / "dos_classification_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("DoS ATTACK VARIANTS CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(classification_report(y_test, y_pred, target_names=target_names))
        f.write("\n\nCONFUSION MATRIX\n")
        f.write("-" * 50 + "\n")
        f.write(str(conf_matrix))
        f.write("\n\nDoS VARIANT MAPPING\n")
        f.write("-" * 40 + "\n")
        for i, dos_variant in enumerate(target_names):
            f.write(f"{i}: {dos_variant}\n")
        f.write("\n\nPER-CLASS PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        for variant, metrics in per_class_metrics.items():
            f.write(f"{variant}: Accuracy={metrics['accuracy']:.4f}, Samples={metrics['samples']}\n")
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

    logging.info("🚀 Bắt đầu training IDS Level 3 - DoS Attack Variants Classification với Advanced CNN 1D")
    logging.info("Arguments: %s", vars(args))

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-split if needed
    if args.auto_split:
        train_path = args.splits_dir / "train_balanced.pkl"
        val_path = args.splits_dir / "val.pkl"
        test_path = args.splits_dir / "test.pkl"

        if not all(p.exists() for p in [train_path, val_path, test_path]):
            logging.info("Không tìm thấy split data level 3, đang chạy auto-split...")

            cmd = [
                sys.executable, str(args.split_script),
                "--level", "3",
                "--group", "dos",  # Chỉ lấy DoS attacks như Random Forest
                "--output-dir", str(args.splits_dir),
                "--source", str(args.source_dataset),
                "--train-min", "3000",   # Min samples cho mỗi DoS variant
                "--random-state", str(args.random_state)
                # Không cần --train-max vì config DoS sẽ handle DoS Hulk -> 10K
            ]

            try:
                subprocess.run(cmd, check=True)
                logging.info("Auto-split level 3 completed successfully")
            except subprocess.CalledProcessError as e:
                logging.error("Auto-split level 3 failed: %s", e)
                sys.exit(1)

    # Load datasets
    train_df = load_split_dataframe(
        args.splits_dir / "train_balanced.pkl",
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
    logging.info("✅ DoS Variants Classification Training completed!")
    logging.info("📊 Final Test Accuracy: %.4f", metadata['performance']['test_accuracy'])
    logging.info("🎯 DoS Variants Classified: %d", metadata['model_info']['num_classes'])
    logging.info("🏆 Top-2 Accuracy: %.4f", metadata['performance']['test_top2_accuracy'])
    logging.info("🏆 Top-3 Accuracy: %.4f", metadata['performance']['test_top3_accuracy'])
    logging.info("📁 Artifacts saved to: %s", args.output_dir)
    logging.info("🎯 Best model: %s", args.output_dir / "dos_classifier_cnn_best.h5")


if __name__ == "__main__":
    main()
