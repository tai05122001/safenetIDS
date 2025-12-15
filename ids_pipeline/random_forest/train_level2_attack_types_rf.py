"""
Script huấn luyện Intrusion Detection Model Level 2 - Attack Types chỉ với Random Forest.

Level 2: Phân loại loại tấn công (dos, ddos, portscan)
Chỉ chạy khi Level 1 = attack

Pipeline chính:
1. Đảm bảo dữ liệu đã được split (tự chạy scripts/split_dataset.py level 2 nếu cần).
2. Đọc các tập train_raw/train_balanced/val/test (chỉ các samples là attack).
3. Sử dụng label_attack_type_encoded (0=dos, 1=ddos, 2=portscan).
4. Huấn luyện Random Forest model.
5. Đánh giá trên validation và holdout/test.
6. Lưu artefact (joblib, metrics, metadata).

Ví dụ chạy:
python ids_pipeline/train_level2_attack_types_rf.py \
    --splits-dir dataset/splits/level2 \
    --train-variant balanced \
    --output-dir artifacts_level2_attack_types_rf
"""
from __future__ import annotations

# ==================== IMPORTS ====================
import argparse
import json
import logging
from typing import Dict, List, Tuple
from pathlib import Path
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd

# Sklearn imports
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


def make_json_safe(value):
    """Chuyển đổi các kiểu numpy thành kiểu Python native để lưu JSON."""
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
    return value


def parse_args() -> argparse.Namespace:
    """Định nghĩa và parse tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình IDS Level 2 - Attack Types với Random Forest."
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("dataset/splits/level2"),
        help="Thư mục chứa các tập dữ liệu đã chia sẵn (mặc định: dataset/splits/level2).",
    )
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=Path("dataset_clean_rf.pkl"),
        help="Dataset nguồn dùng để split level 2 nếu chưa có (mặc định: dataset_clean_rf.pkl).",
    )
    parser.add_argument(
        "--train-variant",
        choices=["raw", "balanced"],
        default="balanced",
        help="Chọn train_raw hay train_balanced để huấn luyện (mặc định: balanced).",
    )
    parser.add_argument(
        "--label-column",
        default="label_attack_type_encoded",
        help="Tên cột nhãn dùng cho training (mặc định: label_attack_type_encoded).",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=["label_group", "label", "label_group_encoded", "label_binary_encoded"],
        help="Danh sách cột bỏ qua khi huấn luyện (tránh data leakage từ Level 1).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="(Tuỳ chọn) tách lại train_raw thành train/test nếu muốn (debug).",
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
        default=Path("artifacts_level2_attack_types_rf"),
        help="Thư mục lưu artefact (mô hình, báo cáo, metadata).",
    )
    parser.add_argument(
        "--auto-split",
        action="store_true",
        default=True,
        help="Tự động chạy split_dataset.py level 2 nếu chưa thấy dữ liệu (mặc định bật).",
    )
    parser.add_argument(
        "--no-auto-split",
        dest="auto_split",
        action="store_false",
        help="Tắt tự động split level 2.",
    )
    parser.add_argument(
        "--split-script",
        type=Path,
        default=Path("scripts/split_dataset.py"),
        help="Đường dẫn script split_dataset.py (mặc định: scripts/split_dataset.py).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Số lượng decision trees trong Random Forest (mặc định: 300).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Độ sâu tối đa của tree (None = không giới hạn, mặc định: None).",
    )
    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=2,
        help="Số mẫu tối thiểu để split node (mặc định: 2).",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Số mẫu tối thiểu ở leaf node (mặc định: 1).",
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


def prepare_features_labels(
    df: pd.DataFrame, label_column: str, drop_columns: List[str]
) -> Tuple[pd.DataFrame, pd.Series, str, List[str]]:
    """Tách features (X) và labels (y) từ DataFrame."""
    column_lookup = {col.lower(): col for col in df.columns}
    
    label_key = label_column.lower()
    if label_key not in column_lookup:
        raise KeyError(f"Không tìm thấy cột nhãn '{label_column}' trong dataset.")
    label_actual = column_lookup[label_key]

    resolved_drop_cols: List[str] = []
    for col in drop_columns:
        key = col.lower()
        if key == label_key:
            continue
        if key in column_lookup:
            resolved_drop_cols.append(column_lookup[key])

    if resolved_drop_cols:
        logging.info("Bỏ các cột không sử dụng: %s", resolved_drop_cols)

    features = df.drop(columns=[label_actual] + resolved_drop_cols, errors="ignore")
    labels = df[label_actual]
    
    if not np.issubdtype(labels.dtype, np.number):
        labels = labels.astype(str)

    logging.info("Sau khi xử lý: %d features.", features.shape[1])
    return features, labels, label_actual, resolved_drop_cols


def build_preprocess_transformer(features: pd.DataFrame) -> ColumnTransformer:
    """
    Tạo preprocessor xử lý cả cột số và cột phân loại.
    
    ⚠️ LƯU Ý QUAN TRỌNG VỀ SCALING:
    - Pipeline này có StandardScaler để scale data khi training
    - Dataset đầu vào (dataset_clean_rf.pkl) đã được scale sẵn (standard scaling)
    - Nếu dataset đã được scale trong preprocess_dataset.py → DOUBLE SCALING → kết quả SAI!
    - → Luôn sử dụng --scale-method none trong preprocess_dataset.py
    """
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [col for col in features.columns if col not in numeric_columns]

    logging.info(
        "Phát hiện %d cột số, %d cột phân loại.",
        len(numeric_columns),
        len(categorical_columns),
    )
    logging.info(
        "⚠️  LƯU Ý: Model pipeline sẽ tự scale data (StandardScaler). "
        "Dataset đầu vào KHÔNG nên được scale sẵn!"
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )


def build_model_pipeline(
    preprocessor: ColumnTransformer,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    y_train: pd.Series | None = None,
) -> Tuple[Pipeline, Dict[int, float] | None]:
    """
    Xây dựng pipeline gồm preprocessor + Random Forest classifier.
    
    Level 2: Attack Types classification
    Classes: 0=dos, 1=ddos, 2=portscan
    """
    # Custom class weights cho attack types
    class_weights = {
        0: 1.5,   # dos
        1: 1.5,   # ddos
        2: 1.5,   # portscan
    }
    
    # Nếu có y_train, tính toán class weights động dựa trên distribution
    if y_train is not None:
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        classes = np.unique(y_train)
        # Tính weights cơ bản từ distribution
        computed_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        computed_dict = dict(zip(classes, computed_weights))
        
        # Điều chỉnh: Tăng thêm weight cho tất cả attack types
        for cls in classes:
            computed_dict[cls] = computed_dict[cls] * 1.2  # Tăng 20%
        
        # Merge với default weights
        for cls, weight in class_weights.items():
            if cls not in computed_dict:
                computed_dict[cls] = weight
        
        class_weights = computed_dict
        logging.info(f"Computed class weights for attack types: {class_weights}")
    else:
        logging.info(f"Using default class weights: {class_weights}")
    
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=42,
        class_weight=class_weights,
    )
    
    logging.info(
        "Random Forest config: n_estimators=%d, max_depth=%s, min_samples_split=%d, min_samples_leaf=%d",
        n_estimators,
        max_depth if max_depth else "None",
        min_samples_split,
        min_samples_leaf,
    )
    logging.info(f"Class weights: {class_weights}")
    
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])
    
    # Trả về cả pipeline và class_weights để lưu vào metadata
    final_weights = class_weights if isinstance(class_weights, dict) else None
    return pipeline, final_weights


def evaluate_model(
    model: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series
) -> Dict[str, Dict[str, float]]:
    """Đánh giá model trên tập dữ liệu evaluation và trả về metrics."""
    logging.info("Đang đánh giá mô hình...")
    y_pred = model.predict(X_eval)
    
    report = classification_report(y_eval, y_pred, output_dict=True, zero_division=0)
    conf_mtx = confusion_matrix(y_eval, y_pred)
    
    logging.info("Classification report:\n%s", json.dumps(report, indent=2))
    logging.info("Confusion matrix:\n%s", conf_mtx)
    
    return {
        "classification_report": report,
        "confusion_matrix": conf_mtx.tolist(),
    }


def save_artifacts(
    model: Pipeline,
    metrics: Dict[str, Dict[str, float]],
    output_dir: Path,
    metadata: Dict[str, str | int | float],
) -> None:
    """Lưu các artifacts: model, metrics, metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "ids_pipeline_level2_attack_types_rf.joblib"
    metrics_path = output_dir / "metrics.json"
    metadata_path = output_dir / "metadata.json"

    joblib.dump(model, model_path)
    logging.info("Đã lưu pipeline vào %s", model_path)

    metrics_path.write_text(json.dumps(make_json_safe(metrics), indent=2), encoding="utf-8")
    logging.info("Đã lưu metrics vào %s", metrics_path)

    metadata_path.write_text(json.dumps(make_json_safe(metadata), indent=2), encoding="utf-8")
    logging.info("Đã lưu metadata vào %s", metadata_path)


def main() -> None:
    """Hàm main: Điểm vào chính của script."""
    args = parse_args()
    setup_logging()

    run_training_pipeline(
        splits_dir=args.splits_dir,
        source_dataset=args.source_dataset,
        auto_split=args.auto_split,
        split_script=args.split_script,
        train_variant=args.train_variant,
        label_column=args.label_column,
        drop_columns=args.drop_columns,
        test_size=args.test_size,
        random_state=args.random_state,
        sample_frac=args.sample_frac,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
    )


def run_training_pipeline(
    *,
    splits_dir: Path | str,
    source_dataset: Path | str,
    auto_split: bool,
    split_script: Path | str,
    train_variant: str = "balanced",
    label_column: str = "label_attack_type_encoded",
    drop_columns: List[str] | None = None,
    test_size: float | None = None,
    random_state: int = 42,
    sample_frac: float | None = None,
    output_dir: Path | str = Path("artifacts_level2_attack_types_rf"),
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
) -> Dict[str, object]:
    """Chạy toàn bộ quy trình huấn luyện level 2 - Attack Types với Random Forest."""
    setup_logging()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    resolved_splits_dir = Path(splits_dir)
    if not resolved_splits_dir.is_absolute():
        resolved_splits_dir = (project_root / resolved_splits_dir).resolve()

    resolved_source_dataset = Path(source_dataset)
    if not resolved_source_dataset.is_absolute():
        resolved_source_dataset = (project_root / resolved_source_dataset).resolve()

    resolved_split_script = Path(split_script)
    if not resolved_split_script.is_absolute():
        resolved_split_script = (project_root / resolved_split_script).resolve()

    resolved_output = Path(output_dir)
    if not resolved_output.is_absolute():
        resolved_output = (project_root / resolved_output).resolve()
    
    effective_drop = drop_columns or []

    if train_variant == "raw":
        train_file = resolved_splits_dir / "train_raw.pkl"
    elif train_variant == "balanced":
        train_file = resolved_splits_dir / "train_balanced.pkl"
    else:
        raise ValueError(f"Loại tập dữ liệu không hợp lệ: {train_variant}")

    logging.info("=" * 80)
    logging.info(f"📁 Train variant: {train_variant}")
    logging.info(f"📁 Train file sẽ được load: {train_file}")
    logging.info(f"📁 File tồn tại: {train_file.exists()}")
    logging.info("=" * 80)

    required_files = [
        train_file,
        resolved_splits_dir / "val.pkl",
        resolved_splits_dir / "test.pkl",
    ]
    
    if auto_split and not all(path.exists() for path in required_files):
        logging.info(
            "Không thấy đủ file split tại %s, gọi split_dataset.py level 2...",
            resolved_splits_dir,
        )
        cmd = [
            sys.executable,
            str(resolved_split_script),
            "--source",
            str(resolved_source_dataset),
            "--level",
            "2",
            "--label-column",
            "label_encoded",
            "--output-dir",
            str(resolved_splits_dir),
            "--train-min",
            str(10_000),
            "--train-max",
            str(200_000),
            "--random-state",
            str(random_state),
        ]
        logging.info("Chạy lệnh: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    df_train = load_split_dataframe(train_file, sample_frac, random_state)
    df_val = load_split_dataframe(
        resolved_splits_dir / "val.pkl", None, random_state
    )
    df_test = load_split_dataframe(
        resolved_splits_dir / "test.pkl", None, random_state
    )

    logging.info("=" * 80)
    logging.info(f"✅ Đã load training data từ: {train_file}")
    logging.info(f"✅ Training data shape: {df_train.shape[0]} rows x {df_train.shape[1]} cols")
    logging.info("=" * 80)

    # Log class distribution trong training data
    if label_column in df_train.columns:
        train_label_counts = df_train[label_column].value_counts().sort_index()
        logging.info("Class distribution in training data (from loaded file):")
        for label, count in train_label_counts.items():
            percentage = (count / len(df_train)) * 100
            logging.info("  Label %s: %d samples (%.2f%%)", label, count, percentage)
        
        # Kiểm tra xem có đúng 3 classes không (0=dos, 1=ddos, 2=portscan)
        unique_classes = sorted(train_label_counts.index)
        expected_classes = [0, 1, 2]
        if not all(cls in unique_classes for cls in expected_classes):
            logging.warning(
                "⚠️  Attack types classification cần 3 classes (0=dos, 1=ddos, 2=portscan). "
                "Phát hiện classes: %s", unique_classes
            )

    df_train_for_model = df_train
    df_holdout = df_test
    
    if test_size is not None and 0 < test_size < 1:
        df_train_for_model, df_holdout = train_test_split(
            df_train,
            test_size=test_size,
            stratify=df_train[label_column],
            random_state=random_state,
        )
        logging.info(
            "Đã tách lại train thành train/test với test_size=%.2f -> train=%d, test=%d",
            test_size,
            df_train_for_model.shape[0],
            df_holdout.shape[0],
        )

    X_train, y_train, label_actual, drop_cols_resolved = prepare_features_labels(
        df_train_for_model, label_column, effective_drop
    )
    X_val, y_val, _, _ = prepare_features_labels(df_val, label_column, effective_drop)
    X_holdout, y_holdout, _, _ = prepare_features_labels(
        df_holdout, label_column, effective_drop
    )
    logging.info("Sử dụng cột nhãn: %s", label_actual)

    preprocessor = build_preprocess_transformer(X_train)
    
    # Log class distribution trong training data
    class_counts = y_train.value_counts().sort_index()
    logging.info("Class distribution in training data:")
    for cls, count in class_counts.items():
        percentage = (count / len(y_train)) * 100
        logging.info(f"  Class {cls}: {count} samples ({percentage:.2f}%)")
    
    pipeline, used_class_weights = build_model_pipeline(
        preprocessor,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        y_train=y_train,
    )

    logging.info(
        "Bắt đầu huấn luyện Random Forest Level 2 - Attack Types (train=%d)...",
        X_train.shape[0]
    )
    pipeline.fit(X_train, y_train)
    logging.info("Huấn luyện hoàn tất.")

    metrics_val = evaluate_model(pipeline, X_val, y_val)
    metrics_holdout = evaluate_model(pipeline, X_holdout, y_holdout)

    metadata = {
        "splits_dir": str(resolved_splits_dir),
        "train_variant": train_variant,
        "train_rows": int(df_train_for_model.shape[0]),
        "val_rows": int(df_val.shape[0]),
        "test_rows": int(df_test.shape[0]),
        "holdout_rows": int(df_holdout.shape[0]),
        "label_column_requested": label_column,
        "label_column_resolved": label_actual,
        "drop_columns_requested": effective_drop,
        "drop_columns_resolved": drop_cols_resolved,
        "test_size_re_split": test_size,
        "random_state": random_state,
        "model_type": "random_forest",
        "level": 2,
        "level_description": "Attack Types (dos, ddos, portscan)",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "class_labels": sorted(y_train.unique()),
        "class_distribution": {int(cls): int(count) for cls, count in class_counts.items()},
        "class_weights": {int(k): float(v) for k, v in used_class_weights.items()} if used_class_weights is not None else None,
        "label_mapping": {
            0: "dos",
            1: "ddos",
            2: "portscan"
        },
    }

    save_artifacts(
        pipeline,
        {"validation": metrics_val, "holdout": metrics_holdout},
        resolved_output,
        metadata,
    )
    logging.info("Pipeline hoàn tất. Artefact lưu tại %s", resolved_output.resolve())

    return {
        "pipeline": pipeline,
        "metrics": {"validation": metrics_val, "holdout": metrics_holdout},
        "metadata": metadata,
        "output_dir": resolved_output.resolve(),
    }


if __name__ == "__main__":
    main()

