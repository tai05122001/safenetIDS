"""
Script huấn luyện Intrusion Detection Model cho dữ liệu level 1 (nhãn nhóm).

Pipeline chính:
1. Đảm bảo dữ liệu đã được split (tự chạy scripts/split_dataset.py nếu cần).
2. Đọc các tập train_raw/train_balanced/val/test.
3. Huấn luyện RandomForestClassifier trong pipeline sklearn.
4. Đánh giá trên validation và holdout/test.
5. Lưu artefact (joblib, metrics, metadata) và (nếu có Keras) export H5.
"""
from __future__ import annotations

import argparse  # Đọc tham số CLI.
import json  # Ghi metrics/metadata dạng JSON.
import logging  # Ghi log quá trình chạy.
from typing import Dict, List, Tuple

from pathlib import Path  # Làm việc với đường dẫn.
import subprocess  # Gọi script split_dataset.py khi cần.
import sys  # Lấy python executable hiện tại.
import joblib  # Lưu/trích xuất pipeline sklearn.
import numpy as np  # Hỗ trợ thao tác số học.
import pandas as pd  # Đọc/ghi DataFrame.
from sklearn.compose import ColumnTransformer  # Xây dựng pipeline tiền xử lý.
from sklearn.ensemble import RandomForestClassifier  # Mô hình chính.
from sklearn.impute import SimpleImputer  # Điền giá trị thiếu.
from sklearn.metrics import classification_report, confusion_matrix  # Metric đánh giá.
from sklearn.model_selection import train_test_split  # Chia train/test.
from sklearn.pipeline import Pipeline  # Kết hợp tiền xử lý + mô hình.
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Chuẩn hóa dữ liệu.


def make_json_safe(value):
    """Chuyển đổi các kiểu numpy thành kiểu Python native để lưu JSON."""
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
    """Định nghĩa và parse tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình IDS level 1 với dữ liệu đã split (train/val/test)."
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
        default=Path("dataset_clean.pkl"),
        help="Dataset nguồn dùng để split level 1 nếu chưa có (mặc định: dataset_clean.pkl).",
    )
    parser.add_argument(
        "--train-variant",
        choices=["raw", "balanced"],
        default="balanced",
        help="Chọn train_raw hay train_balanced để huấn luyện (mặc định: balanced).",
    )
    parser.add_argument(
        "--label-column",
        default="label_group_encoded",
        help="Tên cột nhãn dùng cho training (mặc định: label_group_encoded).",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=["label_group", "label"],
        help="Danh sách cột bỏ qua khi huấn luyện (ví dụ cột label_group, label gốc).",
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
        default=Path("artifacts"),
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
        default=Path("scripts/split_dataset.py"),
        help="Đường dẫn script split_dataset.py (mặc định: scripts/split_dataset.py).",
    )
    return parser.parse_args()


def setup_logging() -> None:
    """Cấu hình logging mức INFO và định dạng thống nhất."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)8s | %(message)s")


def load_split_dataframe(path: Path, sample_frac: float | None, random_state: int) -> pd.DataFrame:
    """Đọc DataFrame từ pickle/CSV và (tuỳ chọn) sample."""
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
    """Trả về X, y sau khi drop các cột không dùng."""
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
    """Tạo preprocessor gồm numeric pipeline + categorical pipeline."""
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [col for col in features.columns if col not in numeric_columns]

    logging.info("Phát hiện %d cột số, %d cột phân loại.", len(numeric_columns), len(categorical_columns))

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
    """Đóng gói preprocessor + RandomForest vào Pipeline sklearn."""
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])


def evaluate_model(
    model: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series
) -> Dict[str, Dict[str, float]]:
    """Chạy predict và trả về báo cáo đánh giá dạng dict."""
    logging.info("Đang đánh giá mô hình...")
    y_pred = model.predict(X_eval)
    report = classification_report(y_eval, y_pred, output_dict=True, zero_division=0)
    conf_mtx = confusion_matrix(y_eval, y_pred)
    logging.info("Classification report:\n%s", json.dumps(report, indent=2))
    logging.info("Confusion matrix:\n%s", conf_mtx)
    return {"classification_report": report, "confusion_matrix": conf_mtx.tolist()}


def save_artifacts(
    model: Pipeline,
    metrics: Dict[str, Dict[str, float]],
    output_dir: Path,
    metadata: Dict[str, str | int | float],
) -> None:
    """Lưu pipeline (joblib), metrics, metadata và optional H5."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "ids_pipeline.joblib"
    metrics_path = output_dir / "metrics.json"
    metadata_path = output_dir / "metadata.json"

    joblib.dump(model, model_path)
    logging.info("Đã lưu pipeline vào %s", model_path)

    try:
        import keras  # Import lazy để tránh phụ thuộc khi không cần H5.
    except ImportError:
        keras = None

    if keras is not None:
        keras_path = output_dir / "ids_pipeline_model_high_level.h5"
        try:
            keras.models.save_model(model, keras_path)
            logging.info("Đã lưu thêm mô hình dạng H5 vào %s", keras_path)
        except Exception as exc:
            logging.warning("Không thể lưu H5: %s", exc)

    metrics_path.write_text(json.dumps(make_json_safe(metrics), indent=2))
    logging.info("Đã lưu metrics vào %s", metrics_path)

    metadata_path.write_text(json.dumps(make_json_safe(metadata), indent=2))
    logging.info("Đã lưu metadata vào %s", metadata_path)


def main() -> None:
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
    )


def run_training_pipeline(
    *,
    splits_dir: Path | str,
    source_dataset: Path | str,
    auto_split: bool,
    split_script: Path | str,
    train_variant: str = "balanced",
    label_column: str = "label_group_encoded",
    drop_columns: List[str] | None = None,
    test_size: float | None = None,
    random_state: int = 42,
    sample_frac: float | None = None,
    output_dir: Path | str = Path("artifacts"),
) -> Dict[str, object]:
    """Chạy toàn bộ quy trình huấn luyện level 1."""
    setup_logging()

    # Chuẩn hóa các đường dẫn và danh sách cột drop.
    script_dir = Path(__file__).resolve().parent  # Thư mục ids_pipeline/.
    project_root = script_dir.parent  # Thư mục gốc dự án.

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
    effective_drop = drop_columns or []  # Bảo vệ trường hợp None.

    # Chọn file train tương ứng với biến thể yêu cầu (raw hoặc balanced).
    if train_variant == "raw":
        train_file = resolved_splits_dir / "train_raw.pkl"
    elif train_variant == "balanced":
        train_file = resolved_splits_dir / "train_balanced.pkl"
    else:
        raise ValueError(f"Loại tập dữ liệu không hợp lệ: {train_variant}")

    # Kiểm tra sự tồn tại của các file quan trọng.
    required_files = [
        train_file,
        resolved_splits_dir / "val.pkl",
        resolved_splits_dir / "test.pkl",
    ]
    # Nếu bật auto_split và thiếu file, gọi split_dataset.py để tạo mới level 1.
    if auto_split and not all(path.exists() for path in required_files):
        logging.info(
            "Không thấy đủ file split tại %s, gọi split_dataset.py level 1...",
            resolved_splits_dir,
        )
        # Chuẩn bị lệnh gọi script split_dataset.py.
        cmd = [
            sys.executable,
            str(resolved_split_script),
            "--source",
            str(resolved_source_dataset),
            "--level",
            "1",
            "--label-column",
            str(label_column),
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

    # Đọc các tập dữ liệu (train/val/test) vừa tạo hoặc có sẵn.
    df_train = load_split_dataframe(train_file, sample_frac, random_state)
    df_val = load_split_dataframe(resolved_splits_dir / "val.pkl", None, random_state)
    df_test = load_split_dataframe(resolved_splits_dir / "test.pkl", None, random_state)

    # Có thể tách lại train thành 2 phần (train + holdout) nếu test_size được thiết lập.
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

    # Tách nhãn và đặc trưng cho từng tập.
    X_train, y_train, label_actual, drop_cols_resolved = prepare_features_labels(
        df_train_for_model, label_column, effective_drop
    )
    X_val, y_val, _, _ = prepare_features_labels(df_val, label_column, effective_drop)
    X_holdout, y_holdout, _, _ = prepare_features_labels(df_holdout, label_column, effective_drop)
    logging.info("Sử dụng cột nhãn: %s", label_actual)

    # Xây dựng pipeline tiền xử lý và mô hình.
    preprocessor = build_preprocess_transformer(X_train)
    pipeline = build_model_pipeline(preprocessor)

    # Huấn luyện mô hình trên tập train chuẩn hóa.
    logging.info("Bắt đầu huấn luyện RandomForest (train=%d)...", X_train.shape[0])
    pipeline.fit(X_train, y_train)
    logging.info("Huấn luyện hoàn tất.")

    # Đánh giá trên tập validation và holdout/test.
    metrics_val = evaluate_model(pipeline, X_val, y_val)
    metrics_holdout = evaluate_model(pipeline, X_holdout, y_holdout)

    # Ghi lại thông tin phục vụ tái hiện thí nghiệm.
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
        "model_type": "RandomForestClassifier",
        "class_labels": sorted(y_train.unique()),
    }

    save_artifacts(
        pipeline,
        {"validation": metrics_val, "holdout": metrics_holdout},
        resolved_output,
        metadata,
    )
    logging.info("Pipeline hoàn tất. Artefact lưu tại %s", resolved_output.resolve())

    # Trả về đối tượng phục vụ notebook/Python API (nếu cần dùng tiếp).
    return {
        "pipeline": pipeline,
        "metrics": {"validation": metrics_val, "holdout": metrics_holdout},
        "metadata": metadata,
        "output_dir": resolved_output.resolve(),
    }


if __name__ == "__main__":
    main()


