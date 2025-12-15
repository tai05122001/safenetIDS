"""
Script huấn luyện Intrusion Detection Model Level 3 - DoS Detail chỉ với Random Forest.

Level 3: Phân loại chi tiết loại DoS (DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest)
Chỉ chạy khi Level 2 = dos

Script đọc dữ liệu đã split trong thư mục level3/<group>/ và huấn luyện Random Forest.

Ví dụ chạy:
python ids_pipeline/train_level3_dos_rf.py \
    --groups dos \
    --splits-dir dataset/splits/level3 \
    --label-column label_encoded \
    --drop-columns label \
    --output-dir artifacts_level3_dos_rf
"""
from __future__ import annotations

# ==================== IMPORTS ====================
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
import subprocess
import sys

# Sklearn imports
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
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
        description="Huấn luyện mô hình IDS Level 3 - DoS Detail với Random Forest cho từng nhóm."
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("dataset/splits/level3"),
        help="Thư mục gốc chứa dữ liệu level 3 (mặc định: dataset/splits/level3/<group>).",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=["dos"],
        help="Danh sách nhóm cần huấn luyện (mặc định: dos). Đã bỏ rare_attack khỏi dataset.",
    )
    parser.add_argument(
        "--train-variant",
        choices=["raw", "balanced"],
        default="balanced",
        help="Chọn tập train để sử dụng (mặc định: balanced).",
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
        default=["label_group", "label", "label_group_encoded", "label_binary_encoded", "label_attack_type_encoded"],
        help="Danh sách các cột bỏ qua khi huấn luyện (tránh data leakage từ Level 1 và Level 2).",
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
        default=Path("artifacts_level3_dos_rf"),
        help="Thư mục lưu artefact (mô hình, báo cáo, metadata).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed tái lập kết quả.",
    )
    parser.add_argument(
        "--auto-split",
        action="store_true",
        default=True,
        help="Tự động chạy split_dataset.py level 3 nếu thiếu dữ liệu (mặc định bật).",
    )
    parser.add_argument(
        "--no-auto-split",
        dest="auto_split",
        action="store_false",
        help="Tắt tự động split level 3.",
    )
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=Path("dataset_clean_rf.pkl"),
        help="Dataset nguồn dùng để split level 3 (mặc định: dataset_clean_rf.pkl).",
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


def load_dataframe(path: Path, sample_frac: float | None, random_state: int) -> pd.DataFrame:
    """Đọc DataFrame từ pickle/CSV và (tuỳ chọn) sample một phần dữ liệu."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset tại {path}")
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Định dạng dữ liệu không được hỗ trợ: {suffix}")

    if sample_frac is not None:
        if not 0 < sample_frac <= 1:
            raise ValueError("--sample-frac phải nằm trong (0, 1].")
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
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

    features = df.drop(columns=[label_actual] + resolved_drop_cols, errors="ignore")
    labels = df[label_actual]
    if not np.issubdtype(labels.dtype, np.number):
        labels = labels.astype(str)
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
        "⚠️  LƯU Ý: Model pipeline sẽ tự scale data (StandardScaler). "
        "Dataset đầu vào KHÔNG nên được scale sẵn!"
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),  # Điền missing values
            ("scaler", StandardScaler()),  # ⚠️ Model sẽ scale data ở đây
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
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
) -> Pipeline:
    """
    Xây dựng pipeline gồm preprocessor + Random Forest classifier.
    
    Sử dụng custom class_weight để cân bằng các class trong Level 2 (chi tiết hơn).
    """
    # Nếu có y_train, tính toán class weights động dựa trên distribution
    if y_train is not None:
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        classes = np.unique(y_train)
        # Tính weights cơ bản từ distribution
        computed_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, computed_weights))
        
        logging.info(f"Computed class weights from training data: {class_weights}")
    else:
        # Fallback: Sử dụng balanced
        class_weights = "balanced"
        logging.info("Using 'balanced' class weights (no training data provided)")
    
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=42,
        class_weight=class_weights,  # Sử dụng computed weights hoặc "balanced"
    )
    
    logging.info(
        "Random Forest config: n_estimators=%d, max_depth=%s, min_samples_split=%d, min_samples_leaf=%d",
        n_estimators,
        max_depth if max_depth else "None",
        min_samples_split,
        min_samples_leaf,
    )
    if isinstance(class_weights, dict):
        logging.info(f"Class weights: {class_weights}")
    
    return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])


def evaluate_model(
    model: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series, tag: str
) -> Dict[str, Dict[str, float]]:
    """Đánh giá model trên tập dữ liệu evaluation và trả về metrics."""
    logging.info("Đang đánh giá trên %s...", tag)
    y_pred = model.predict(X_eval)
    report = classification_report(y_eval, y_pred, output_dict=True, zero_division=0)
    conf_mtx = confusion_matrix(y_eval, y_pred)
    logging.info("%s classification report:\n%s", tag, json.dumps(report, indent=2))
    logging.info("%s confusion matrix:\n%s", tag, conf_mtx)
    return {"classification_report": report, "confusion_matrix": conf_mtx.tolist()}


def save_artifacts(
    model: Pipeline,
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path,
    metadata: Dict[str, object],
) -> None:
    """Lưu các artifacts: model, metrics, metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "ids_pipeline_level3_dos_rf.joblib"
    metrics_path = output_dir / "metrics.json"
    metadata_path = output_dir / "metadata.json"

    joblib.dump(model, model_path)
    logging.info("Đã lưu pipeline vào %s", model_path)

    metrics_path.write_text(json.dumps(make_json_safe(metrics), indent=2), encoding="utf-8")
    logging.info("Đã lưu metrics vào %s", metrics_path)

    metadata_path.write_text(json.dumps(make_json_safe(metadata), indent=2), encoding="utf-8")
    logging.info("Đã lưu metadata vào %s", metadata_path)


def run_training_pipeline(
    *,
    group: str,
    splits_dir: Path | str,
    source_dataset: Path | str,
    auto_split: bool,
    split_script: Path | str,
    train_variant: str,
    label_column: str,
    drop_columns: List[str] | None,
    sample_frac: float | None,
    output_dir: Path | str,
    random_state: int,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
) -> Dict[str, object]:
    """Chạy toàn bộ quy trình huấn luyện level 3 - DoS Detail với Random Forest cho một nhóm."""
    setup_logging()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    splits_path = Path(splits_dir)
    if not splits_path.is_absolute():
        splits_path = (project_root / splits_path).resolve()

    def normalize_group(name: str) -> str:
        normalized = name.strip().lower().replace("-", "_")
        # Đã bỏ rare_attack khỏi dataset, không cần normalize nữa
        return normalized

    normalized_group = normalize_group(group)

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

    train_file = splits_path / f"train_{train_variant}.pkl"
    val_file = splits_path / "val.pkl"
    test_file = splits_path / "test.pkl"

    required_files = [train_file, val_file, test_file]
    if auto_split and not all(path.exists() for path in required_files):
        logging.info(
            "[%s] Không thấy đủ file split, chạy split_dataset.py level 3...",
            normalized_group,
        )
        cmd = [
            sys.executable,
            str(resolved_split_script),
            "--source",
            str(resolved_source_dataset),
            "--level",
            "3",
            "--group",
            normalized_group,
            "--label-column",
            label_column,
            "--output-dir",
            str(splits_path),
            "--train-min",
            "10000",
            "--train-max",
            "200000",
            "--random-state",
            str(random_state),
        ]
        logging.info("[%s] Chạy lệnh: %s", normalized_group, " ".join(cmd))
        subprocess.run(cmd, check=True)

    df_train = load_dataframe(train_file, sample_frac, random_state)
    df_val = load_dataframe(val_file, None, random_state)
    df_test = load_dataframe(test_file, None, random_state)

    # Log class distribution trong training data TRƯỚC khi prepare features
    if label_column in df_train.columns:
        train_label_counts = df_train[label_column].value_counts().sort_index()
        logging.info("[%s] Class distribution in training data (from split):", normalized_group)
        for label, count in train_label_counts.items():
            percentage = (count / len(df_train)) * 100
            logging.info("[%s]   %s: %d samples (%.2f%%)", normalized_group, label, count, percentage)
        
        # Kiểm tra xem DoS Hulk đã được giảm xuống 30000 chưa (nếu là group dos)
        if normalized_group == "dos":
            hulk_labels = [l for l in train_label_counts.index if 'hulk' in str(l).lower()]
            if hulk_labels:
                hulk_count = sum(train_label_counts[l] for l in hulk_labels)
                if hulk_count > 35000:  # Nếu vẫn còn > 35K thì có thể chưa được balance
                    logging.warning(
                        "[%s] DoS Hulk có %d samples - có thể chưa được giảm xuống 30000. "
                        "Kiểm tra lại split_dataset.py có áp dụng balance config không.",
                        normalized_group, hulk_count
                    )
                else:
                    logging.info(
                        "[%s] ✓ DoS Hulk đã được giảm xuống %d samples (target: 30000)",
                        normalized_group, hulk_count
                    )

    X_train, y_train, label_actual, drop_cols_resolved = prepare_features_labels(
        df_train, label_column, effective_drop
    )
    X_val, y_val, _, _ = prepare_features_labels(df_val, label_column, effective_drop)
    X_test, y_test, _, _ = prepare_features_labels(df_test, label_column, effective_drop)
    logging.info("Sử dụng cột nhãn: %s", label_actual)

    preprocessor = build_preprocess_transformer(X_train)
    
    # Log class distribution trong training data
    class_counts = y_train.value_counts().sort_index()
    logging.info("[%s] Class distribution in training data:", normalized_group)
    for cls, count in class_counts.items():
        percentage = (count / len(y_train)) * 100
        logging.info("[%s]   Class %s: %d samples (%.2f%%)", normalized_group, cls, count, percentage)
    
    pipeline = build_model_pipeline(
        preprocessor,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        y_train=y_train,  # Truyền y_train để tính class weights động
    )

    logging.info(
        "Bắt đầu huấn luyện Random Forest (group=%s, train=%d)...",
        normalized_group,
        X_train.shape[0],
    )
    pipeline.fit(X_train, y_train)
    logging.info("Huấn luyện hoàn tất.")

    metrics_val = evaluate_model(pipeline, X_val, y_val, tag="validation")
    metrics_test = evaluate_model(pipeline, X_test, y_test, tag="test")

    # Tạo label_mapping từ encoded values sang label names
    # Tìm cột label gốc (không phải _encoded) để map
    original_label_col = None
    possible_label_cols = ['label', 'Label']
    for col in possible_label_cols:
        if col in df_train.columns and col != label_actual:
            original_label_col = col
            break
    
    # Nếu không tìm thấy, thử tìm cột có chứa 'label' nhưng không phải encoded
    if original_label_col is None:
        for col in df_train.columns:
            if 'label' in col.lower() and '_encoded' not in col.lower() and col != label_actual:
                original_label_col = col
                break
    
    # Tạo mapping từ encoded values -> label names
    label_mapping = {}
    if original_label_col and original_label_col in df_train.columns:
        for encoded_val in sorted(y_train.unique()):
            # Lấy một sample với encoded value này
            sample = df_train[df_train[label_actual] == encoded_val].iloc[0]
            label_name = str(sample[original_label_col]).strip()
            label_mapping[int(encoded_val)] = label_name
    else:
        # Fallback: Dùng class_labels nếu không tìm thấy label gốc
        # Mapping sẽ là encoded value -> encoded value (không có tên)
        for encoded_val in sorted(y_train.unique()):
            label_mapping[int(encoded_val)] = f"DoS_type_{int(encoded_val)}"
        logging.warning("Không tìm thấy cột label gốc để tạo label_mapping. Sử dụng fallback mapping.")

    metadata = {
        "group": group,
        "normalized_group": normalized_group,
        "splits_dir": str(splits_path),
        "train_variant": train_variant,
        "train_rows": int(df_train.shape[0]),
        "val_rows": int(df_val.shape[0]),
        "test_rows": int(df_test.shape[0]),
        "label_column": label_actual,
        "drop_columns_requested": effective_drop,
        "drop_columns_resolved": drop_cols_resolved,
        "random_state": random_state,
        "model_type": "random_forest",
        "level": 3,
        "level_description": "DoS Detail (DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest)",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "class_labels": sorted(y_train.unique()),
        "class_distribution": {int(cls): int(count) for cls, count in class_counts.items()},
        "class_weights": {int(k): float(v) for k, v in pipeline.named_steps['classifier'].class_weight_.items()} if hasattr(pipeline.named_steps['classifier'], 'class_weight_') else None,
        "label_mapping": label_mapping,
    }

    save_artifacts(
        pipeline,
        {"validation": metrics_val, "test": metrics_test},
        resolved_output,
        metadata,
    )

    return {
        "pipeline": pipeline,
        "metrics": {"validation": metrics_val, "test": metrics_test},
        "metadata": metadata,
        "output_dir": resolved_output.resolve(),
    }


def main() -> None:
    """Hàm main: Điểm vào chính của script."""
    args = parse_args()
    setup_logging()
    
    # Huấn luyện cho từng nhóm trong danh sách
    for group in args.groups:
        group = group.lower()
        splits_dir = Path(args.splits_dir) / group
        output_dir = Path(args.output_dir) / group
        logging.info("=== Huấn luyện Level 3 - DoS Detail (Random Forest) cho nhóm: %s ===", group)
        run_training_pipeline(
            group=group,
            splits_dir=splits_dir,
            source_dataset=args.source_dataset,
            auto_split=args.auto_split,
            split_script=args.split_script,
            train_variant=args.train_variant,
            label_column=args.label_column,
            drop_columns=args.drop_columns,
            sample_frac=args.sample_frac,
            output_dir=output_dir,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
        )


if __name__ == "__main__":
    main()

