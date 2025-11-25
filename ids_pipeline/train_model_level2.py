"""
Huấn luyện mô hình IDS Level 2 cho từng nhóm (ví dụ: DOS, RareAttack, PortScan).
Script đọc dữ liệu đã split trong thư mục level2/<group>/ và huấn luyện mô hình con phân biệt nhãn chi tiết.

Hỗ trợ nhiều loại model: Random Forest, XGBoost, LightGBM, hoặc Ensemble Voting.
Ensemble Voting: Kết hợp nhiều model, dự đoán label được vote nhiều nhất.

Ví dụ chạy:
python ids_pipeline/train_model_level2.py \
    --groups dos rare_attack portscan \
    --model-type ensemble \
    --splits-dir dataset/splits/level2 \
    --label-column label_encoded \
    --drop-columns label \
    --output-dir artifacts_level2
"""
from __future__ import annotations  # Cho phép dùng type hints mới

# ==================== IMPORTS ====================
import argparse  # Đọc và parse tham số từ dòng lệnh
import json  # Ghi/đọc dữ liệu dạng JSON
import logging  # Ghi log quá trình chạy
from pathlib import Path  # Làm việc với đường dẫn file/thư mục
from typing import Dict, List, Tuple  # Type hints

import joblib  # Lưu/load pipeline sklearn
import numpy as np  # Hỗ trợ thao tác số học
import pandas as pd  # Đọc/ghi DataFrame
import subprocess  # Gọi script split_dataset.py khi cần
import sys  # Lấy python executable hiện tại

# Sklearn imports cho preprocessing và pipeline
from sklearn.compose import ColumnTransformer  # Pipeline tiền xử lý cho nhiều loại cột
from sklearn.impute import SimpleImputer  # Điền giá trị thiếu
from sklearn.metrics import classification_report, confusion_matrix  # Metric đánh giá
from sklearn.pipeline import Pipeline  # Kết hợp tiền xử lý + mô hình
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Mã hóa và chuẩn hóa

# Sklearn imports cho các model classification
from sklearn.ensemble import (
    RandomForestClassifier,  # Random Forest
    ExtraTreesClassifier,  # Extra Trees
    VotingClassifier,  # Ensemble voting - kết hợp nhiều model
)

# Gradient Boosting models (cần cài: pip install xgboost lightgbm)
try:
    from xgboost import XGBClassifier  # XGBoost
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier  # LightGBM
except ImportError:
    LGBMClassifier = None


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
        default=["label_group", "label"],
        help="Danh sách các cột bỏ qua khi huấn luyện (ví dụ: label_group, label gốc).",
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
    parser.add_argument(
        "--model-type",
        choices=["random_forest", "xgboost", "lightgbm", "ensemble", "voting"],
        default="ensemble",
        help=(
            "Loại model sử dụng: "
            "random_forest (Random Forest đơn), "
            "xgboost (XGBoost đơn), "
            "lightgbm (LightGBM đơn), "
            "ensemble/voting (kết hợp nhiều model, chọn label được vote nhiều nhất). "
            "Mặc định: ensemble."
        ),
    )
    return parser.parse_args()  # Parse và trả về namespace


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


def build_model_pipeline(
    preprocessor: ColumnTransformer, model_type: str = "ensemble"
) -> Pipeline:
    """
    Xây dựng pipeline gồm preprocessor + classifier.
    
    Args:
        preprocessor: ColumnTransformer đã được fit với dữ liệu train
        model_type: Loại model sử dụng (random_forest, xgboost, lightgbm, ensemble)
    
    Returns:
        Pipeline sklearn kết hợp preprocessing + classification
    """
    # ========== RANDOM FOREST ==========
    if model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=300,  # Số lượng decision trees
            max_depth=None,  # Không giới hạn độ sâu
            n_jobs=-1,  # Sử dụng tất cả CPU cores
            random_state=42,  # Seed để tái lập
            class_weight="balanced_subsample",  # Cân bằng class weights
        )
        return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])
    
    # ========== XGBOOST ==========
    elif model_type == "xgboost":
        if XGBClassifier is None:
            raise ImportError("XGBoost chưa được cài đặt. Chạy: pip install xgboost")
        classifier = XGBClassifier(
            n_estimators=300,  # Số lượng boosting rounds
            max_depth=6,  # Độ sâu tối đa
            learning_rate=0.1,  # Tốc độ học
            random_state=42,  # Seed
            n_jobs=-1,  # Parallel processing
            eval_metric="mlogloss",  # Metric đánh giá
        )
        return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])
    
    # ========== LIGHTGBM ==========
    elif model_type == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("LightGBM chưa được cài đặt. Chạy: pip install lightgbm")
        classifier = LGBMClassifier(
            n_estimators=300,  # Số lượng boosting rounds
            max_depth=6,  # Độ sâu tối đa
            learning_rate=0.1,  # Tốc độ học
            random_state=42,  # Seed
            n_jobs=-1,  # Parallel processing
            class_weight="balanced",  # Cân bằng class weights
        )
        return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])
    
    # ========== ENSEMBLE VOTING ==========
    elif model_type in ["ensemble", "voting"]:
        # Kết hợp nhiều model, label được vote nhiều nhất sẽ là kết quả
        estimators = []
        
        # 1. Random Forest
        estimators.append(
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=42,
                    class_weight="balanced_subsample",
                ),
            )
        )
        
        # 2. Extra Trees
        estimators.append(
            (
                "et",
                ExtraTreesClassifier(
                    n_estimators=200,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=42,
                    class_weight="balanced_subsample",
                ),
            )
        )
        
        # 3. XGBoost
        if XGBClassifier is not None:
            estimators.append(
                (
                    "xgb",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        n_jobs=-1,
                        eval_metric="mlogloss",
                    ),
                )
            )
        else:
            logging.warning("XGBoost chưa được cài, bỏ qua trong ensemble")
        
        # 4. LightGBM
        if LGBMClassifier is not None:
            estimators.append(
                (
                    "lgbm",
                    LGBMClassifier(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        n_jobs=-1,
                        class_weight="balanced",
                    ),
                )
            )
        else:
            logging.warning("LightGBM chưa được cài, bỏ qua trong ensemble")
        
        if len(estimators) < 2:
            raise ValueError(
                "Cần ít nhất 2 models để ensemble. "
                "Hãy cài đặt XGBoost và/hoặc LightGBM: pip install xgboost lightgbm"
            )
        
        # VotingClassifier: majority vote (hard voting)
        classifier = VotingClassifier(
            estimators=estimators,  # Danh sách các models
            voting="hard",  # Chọn label được vote nhiều nhất
            n_jobs=-1,  # Parallel processing
        )
        
        logging.info(
            "Sử dụng Ensemble Voting với %d models: %s",
            len(estimators),
            [name for name, _ in estimators],
        )
        return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])
    
    else:
        raise ValueError(
            f"Model type không hợp lệ: {model_type}. "
            "Chọn một trong: random_forest, xgboost, lightgbm, ensemble, voting"
        )


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
    group: str,  # Tên nhóm (dos, rare_attack, portscan, etc.)
    splits_dir: Path | str,  # Thư mục chứa splits
    source_dataset: Path | str,  # Dataset nguồn
    auto_split: bool,  # Tự động split nếu cần
    split_script: Path | str,  # Đường dẫn script split
    train_variant: str,  # train_raw hay train_balanced
    label_column: str,  # Tên cột label
    drop_columns: List[str] | None,  # Các cột cần bỏ qua
    sample_frac: float | None,  # Tỷ lệ sample
    output_dir: Path | str,  # Thư mục lưu artifacts
    random_state: int,  # Seed
    model_type: str = "ensemble",  # Loại model sử dụng
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

    # Xây dựng pipeline tiền xử lý và mô hình
    preprocessor = build_preprocess_transformer(X_train)
    pipeline = build_model_pipeline(preprocessor, model_type=model_type)

    logging.info(
        "Bắt đầu huấn luyện %s (group=%s, train=%d)...",
        model_type.upper(),
        normalized_group,
        X_train.shape[0],
    )
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
        "model_type": model_type,  # Lưu loại model đã sử dụng
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
    """
    Hàm main: Điểm vào chính của script.
    
    Huấn luyện model level 2 cho từng nhóm trong danh sách groups.
    """
    args = parse_args()
    setup_logging()  # Setup logging
    
    # Huấn luyện cho từng nhóm trong danh sách
    for group in args.groups:
        group = group.lower()  # Chuẩn hóa tên nhóm về lowercase
        splits_dir = Path(args.splits_dir) / group  # Thư mục splits cho nhóm này
        output_dir = Path(args.output_dir) / group  # Thư mục output cho nhóm này
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
            model_type=args.model_type,  # Loại model sử dụng
        )


if __name__ == "__main__":
    main()
