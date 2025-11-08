"""
Huấn luyện mô hình IDS Level 2 cho từng nhóm (ví dụ: DOS, RareAttack).
Script đọc dữ liệu đã split trong thư mục level2/<group>/ và huấn luyện mô hình con phân biệt nhãn chi tiết.

Ví dụ chạy:
python ids_pipeline/train_model_level2.py \
    --group dos \
    --splits-dir dataset/splits_hierarchical/level2/dos \
    --label-column label_encoded \
    --drop-columns label \
    --output-dir outputs/level2/dos
"""
from __future__ import annotations

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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình IDS Level 2 cho từng nhóm (DOS, RareAttack, ...)."
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("dataset/splits/level2"),
        help="Thư mục gốc chứa dữ liệu level 2 (mặc định: dataset/splits/level2/<group>).",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=["dos", "rare_attack"],
        help="Danh sách nhóm cần huấn luyện (mặc định: dos rare_attack).",
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
        default=["label"],
        help="Danh sách các cột bỏ qua khi huấn luyện (ví dụ: label gốc).",
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
        default=Path("artifacts_level2"),
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
        help="Tự động chạy split_dataset.py level 2 nếu thiếu dữ liệu (mặc định bật).",
    )
    parser.add_argument(
        "--no-auto-split",
        dest="auto_split",
        action="store_false",
        help="Tắt tự động split level 2.",
    )
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=Path("dataset_clean.pkl"),
        help="Dataset nguồn dùng để split level 2 (mặc định: dataset_clean.pkl).",
    )
    parser.add_argument(
        "--split-script",
        type=Path,
        default=Path("scripts/split_dataset.py"),
        help="Đường dẫn script split_dataset.py (mặc định: scripts/split_dataset.py).",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)8s | %(message)s",
    )


def load_dataframe(path: Path, sample_frac: float | None, random_state: int) -> pd.DataFrame:
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
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [col for col in features.columns if col not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
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


def build_model_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])


def evaluate_model(
    model: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series, tag: str
) -> Dict[str, Dict[str, float]]:
    logging.info("Đánh giá trên %s...", tag)
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
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "ids_pipeline_level2.joblib"
    metrics_path = output_dir / "metrics.json"
    metadata_path = output_dir / "metadata.json"

    joblib.dump(model, model_path)
    logging.info("Đã lưu pipeline vào %s", model_path)

    try:
        import keras
    except ImportError:
        keras = None

    if keras is not None:
        keras_path = output_dir / "ids_pipeline_level2_model.h5"
        try:
            keras.models.save_model(model, keras_path)
            logging.info("Đã lưu thêm mô hình dạng H5 vào %s", keras_path)
        except Exception as exc:
            logging.warning("Không thể lưu H5: %s", exc)

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
) -> Dict[str, object]:
    setup_logging()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    splits_path = Path(splits_dir)
    if not splits_path.is_absolute():
        splits_path = (project_root / splits_path).resolve()

    def normalize_group(name: str) -> str:
        normalized = name.strip().lower().replace("-", "_")
        if normalized in {"rare_attack", "rareattack", "rareattack", "rare_attack"}:
            normalized = "rare_attack"
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
            "[%s] Không thấy đủ file split, chạy split_dataset.py level 2...",
            normalized_group,
        )
        cmd = [
            sys.executable,
            str(resolved_split_script),
            "--source",
            str(resolved_source_dataset),
            "--level",
            "2",
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

    X_train, y_train, label_actual, drop_cols_resolved = prepare_features_labels(
        df_train, label_column, effective_drop
    )
    X_val, y_val, _, _ = prepare_features_labels(df_val, label_column, effective_drop)
    X_test, y_test, _, _ = prepare_features_labels(df_test, label_column, effective_drop)
    logging.info("Sử dụng cột nhãn: %s", label_actual)

    preprocessor = build_preprocess_transformer(X_train)
    pipeline = build_model_pipeline(preprocessor)

    logging.info("Bắt đầu huấn luyện RandomForest (group=%s, train=%d)...", normalized_group, X_train.shape[0])
    pipeline.fit(X_train, y_train)
    logging.info("Huấn luyện hoàn tất.")

    metrics_val = evaluate_model(pipeline, X_val, y_val, tag="validation")
    metrics_test = evaluate_model(pipeline, X_test, y_test, tag="test")

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
        "model_type": "RandomForestClassifier",
        "class_labels": sorted(y_train.unique()),
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
    args = parse_args()
    for group in args.groups:
        group = group.lower()
        splits_dir = Path(args.splits_dir) / group
        output_dir = Path(args.output_dir) / group
        logging.info("=== Huấn luyện Level 2 cho nhóm: %s ===", group)
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
        )


if __name__ == "__main__":
    main()
