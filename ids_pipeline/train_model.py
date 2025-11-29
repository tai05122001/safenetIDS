"""
Script huấn luyện Intrusion Detection Model cho dữ liệu level 1 (nhãn nhóm).

Pipeline chính:
1. Đảm bảo dữ liệu đã được split (tự chạy scripts/split_dataset.py nếu cần).
2. Đọc các tập train_raw/train_balanced/val/test.
3. Huấn luyện model (RandomForest, XGBoost, LightGBM, hoặc Ensemble Voting).
4. Đánh giá trên validation và holdout/test.
5. Lưu artefact (joblib, metrics, metadata) và (nếu có Keras) export H5.

Ensemble Voting: Kết hợp nhiều model, dự đoán label được vote nhiều nhất.
"""
from __future__ import annotations  # Cho phép dùng type hints mới (Python 3.7+)

# ==================== IMPORTS ====================
import argparse  # Đọc và parse tham số từ dòng lệnh (CLI arguments)
import json  # Ghi/đọc dữ liệu dạng JSON (metrics, metadata)
import logging  # Ghi log quá trình chạy để debug và theo dõi
from typing import Dict, List, Tuple  # Type hints cho dict, list, tuple

from pathlib import Path  # Làm việc với đường dẫn file/thư mục (cross-platform)
import subprocess  # Gọi script split_dataset.py khi cần tự động split
import sys  # Lấy python executable hiện tại để chạy subprocess
import joblib  # Lưu/load pipeline sklearn (model persistence)
import numpy as np  # Hỗ trợ thao tác số học, array operations
import pandas as pd  # Đọc/ghi DataFrame, xử lý dữ liệu dạng bảng

# Sklearn imports cho preprocessing và pipeline
from sklearn.compose import ColumnTransformer  # Xây dựng pipeline tiền xử lý cho nhiều loại cột
from sklearn.impute import SimpleImputer  # Điền giá trị thiếu (missing values)
from sklearn.metrics import classification_report, confusion_matrix  # Metric đánh giá model
from sklearn.model_selection import train_test_split  # Chia dữ liệu train/test
from sklearn.pipeline import Pipeline  # Kết hợp tiền xử lý + mô hình thành pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Mã hóa và chuẩn hóa dữ liệu

# Sklearn imports cho các model classification
from sklearn.ensemble import (
    RandomForestClassifier,  # Random Forest - ensemble của decision trees
    ExtraTreesClassifier,  # Extra Trees - biến thể của RF với randomness cao hơn
    VotingClassifier,  # Ensemble voting - kết hợp nhiều model, chọn label được vote nhiều nhất
    GradientBoostingClassifier,  # Gradient Boosting (sklearn native)
    AdaBoostClassifier,  # Adaptive Boosting
    BaggingClassifier,  # Bagging ensemble
)
from sklearn.tree import DecisionTreeClassifier  # Decision Tree đơn
from sklearn.linear_model import LogisticRegression  # Logistic Regression
from sklearn.neural_network import MLPClassifier  # Multi-layer Perceptron (Neural Network)
from sklearn.svm import SVC  # Support Vector Classifier

# Gradient Boosting models (cần cài: pip install xgboost lightgbm)
try:
    from xgboost import XGBClassifier  # XGBoost - gradient boosting mạnh mẽ
except ImportError:
    XGBClassifier = None  # Nếu chưa cài thì set None

try:
    from lightgbm import LGBMClassifier  # LightGBM - gradient boosting nhanh và hiệu quả
except ImportError:
    LGBMClassifier = None  # Nếu chưa cài thì set None

# CatBoost (optional - cần cài: pip install catboost)
try:
    from catboost import CatBoostClassifier  # CatBoost - gradient boosting với categorical support
except ImportError:
    CatBoostClassifier = None  # Nếu chưa cài thì set None


def make_json_safe(value):
    """
    Chuyển đổi các kiểu numpy thành kiểu Python native để lưu JSON.
    
    JSON không hỗ trợ numpy types, nên cần convert sang Python native types.
    Hàm này đệ quy xử lý dict, list, tuple, numpy arrays, numpy numbers.
    
    Args:
        value: Giá trị cần convert (có thể là dict, list, tuple, numpy type)
    
    Returns:
        Giá trị đã được convert sang Python native types
    """
    # Nếu là dictionary, đệ quy convert từng key-value pair
    if isinstance(value, dict):
        return {make_json_safe(k): make_json_safe(v) for k, v in value.items()}
    # Nếu là list, đệ quy convert từng phần tử
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    # Nếu là tuple, convert sang list rồi đệ quy
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    # Nếu là numpy integer, convert sang Python int
    if isinstance(value, (np.integer,)):
        return int(value)
    # Nếu là numpy float, convert sang Python float
    if isinstance(value, (np.floating,)):
        return float(value)
    # Nếu là numpy array, convert sang list rồi đệ quy
    if isinstance(value, (np.ndarray,)):
        return make_json_safe(value.tolist())
    # Các kiểu khác giữ nguyên (string, None, bool, etc.)
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
    return parser.parse_args()  # Parse và trả về namespace chứa tất cả arguments


def setup_logging() -> None:
    """
    Cấu hình logging mức INFO và định dạng thống nhất.
    
    Thiết lập format log: [timestamp] | [level] | [message]
    Mức INFO sẽ hiển thị các thông tin quan trọng trong quá trình training.
    """
    logging.basicConfig(
        level=logging.INFO,  # Chỉ hiển thị log từ mức INFO trở lên (INFO, WARNING, ERROR)
        format="%(asctime)s | %(levelname)8s | %(message)s",  # Format: timestamp | level | message
    )


def load_split_dataframe(
    path: Path, sample_frac: float | None, random_state: int
) -> pd.DataFrame:
    """
    Đọc DataFrame từ pickle/CSV và (tuỳ chọn) sample một phần dữ liệu.
    
    Args:
        path: Đường dẫn đến file dữ liệu (pickle hoặc CSV)
        sample_frac: Tỷ lệ sample (0 < frac <= 1), None = không sample
        random_state: Seed để tái lập kết quả sampling
    
    Returns:
        DataFrame đã được đọc và (có thể) đã được sample
    """
    # Kiểm tra file có tồn tại không
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {path}")

    logging.info("Đang đọc dữ liệu từ %s", path)
    # Lấy extension của file để xác định định dạng
    suffix = path.suffix.lower()
    
    # Đọc file pickle (nhanh hơn CSV, giữ nguyên kiểu dữ liệu)
    if suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    # Đọc file CSV (chậm hơn nhưng dễ xem)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Định dạng dữ liệu không được hỗ trợ: {suffix}")

    # Kiểm tra kết quả đọc có phải DataFrame không
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Dataset phải là pandas DataFrame sau khi đọc.")

    # Nếu có yêu cầu sample, lấy một phần dữ liệu
    if sample_frac is not None:
        # Validate sample_frac phải trong khoảng (0, 1]
        if not 0 < sample_frac <= 1:
            raise ValueError("--sample-frac phải nằm trong (0, 1].")
        # Sample ngẫu nhiên với tỷ lệ sample_frac
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
        logging.info("Sample %.2f => %d rows.", sample_frac, df.shape[0])
    else:
        # Không sample, log thông tin dataset
        logging.info("Dataset có %d dòng, %d cột.", df.shape[0], df.shape[1])
    return df


def prepare_features_labels(
    df: pd.DataFrame, label_column: str, drop_columns: List[str]
) -> Tuple[pd.DataFrame, pd.Series, str, List[str]]:
    """
    Tách features (X) và labels (y) từ DataFrame, loại bỏ các cột không cần thiết.
    
    Args:
        df: DataFrame chứa dữ liệu đầy đủ
        label_column: Tên cột chứa labels (có thể không phân biệt hoa thường)
        drop_columns: Danh sách cột cần loại bỏ (không dùng để train)
    
    Returns:
        Tuple gồm:
        - features: DataFrame chỉ chứa features (X)
        - labels: Series chứa labels (y)
        - label_actual: Tên cột label thực tế trong DataFrame (sau khi resolve)
        - resolved_drop_cols: Danh sách cột đã được loại bỏ
    """
    # Tạo lookup dictionary: lowercase column name -> actual column name
    # Để tìm cột không phân biệt hoa thường
    column_lookup = {col.lower(): col for col in df.columns}
    
    # Tìm cột label (không phân biệt hoa thường)
    label_key = label_column.lower()
    if label_key not in column_lookup:
        raise KeyError(f"Không tìm thấy cột nhãn '{label_column}' trong dataset.")
    label_actual = column_lookup[label_key]  # Tên cột label thực tế

    # Resolve các cột cần drop (tìm tên thực tế trong DataFrame)
    resolved_drop_cols: List[str] = []
    for col in drop_columns:
        key = col.lower()  # Chuyển sang lowercase để so sánh
        # Bỏ qua nếu trùng với label column (sẽ drop riêng)
        if key == label_key:
            continue
        # Nếu tìm thấy trong lookup, thêm vào danh sách drop
        if key in column_lookup:
            resolved_drop_cols.append(column_lookup[key])

    # Log các cột sẽ được loại bỏ
    if resolved_drop_cols:
        logging.info("Bỏ các cột không sử dụng: %s", resolved_drop_cols)

    # Tách features: loại bỏ label column và các cột trong drop list
    features = df.drop(columns=[label_actual] + resolved_drop_cols, errors="ignore")
    # Tách labels: lấy cột label
    labels = df[label_actual]
    
    # Đảm bảo labels là kiểu số hoặc string (không phải object phức tạp)
    if not np.issubdtype(labels.dtype, np.number):
        labels = labels.astype(str)

    logging.info("Sau khi xử lý: %d features.", features.shape[1])
    return features, labels, label_actual, resolved_drop_cols


def build_preprocess_transformer(features: pd.DataFrame) -> ColumnTransformer:
    """
    Tạo preprocessor xử lý cả cột số và cột phân loại.
    
    Pipeline gồm:
    - Numeric columns: Impute missing values (median) -> Standardize (z-score)
    - Categorical columns: Impute missing values (mode) -> One-hot encoding
    
    Args:
        features: DataFrame chứa features (chưa xử lý)
    
    Returns:
        ColumnTransformer đã được cấu hình để xử lý cả numeric và categorical columns
    """
    # Tách các cột số (numeric) và cột phân loại (categorical)
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [col for col in features.columns if col not in numeric_columns]

    logging.info(
        "Phát hiện %d cột số, %d cột phân loại.",
        len(numeric_columns),
        len(categorical_columns),
    )

    # Pipeline cho cột số:
    # 1. Imputer: Điền giá trị thiếu bằng median (ổn định hơn mean với outliers)
    # 2. Scaler: Chuẩn hóa về mean=0, std=1 (z-score normalization)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),  # Điền missing bằng median
            ("scaler", StandardScaler()),  # Chuẩn hóa: (x - mean) / std
        ]
    )
    
    # Pipeline cho cột phân loại:
    # 1. Imputer: Điền giá trị thiếu bằng mode (giá trị xuất hiện nhiều nhất)
    # 2. Encoder: One-hot encoding (chuyển categorical thành binary columns)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),  # Điền missing bằng mode
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",  # Bỏ qua giá trị chưa thấy khi predict
                    sparse_output=False,  # Trả về dense array (không phải sparse matrix)
                ),
            ),
        ]
    )
    
    # ColumnTransformer: Áp dụng pipeline khác nhau cho từng loại cột
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),  # Áp dụng numeric_pipeline cho numeric columns
            ("cat", categorical_pipeline, categorical_columns),  # Áp dụng categorical_pipeline cho categorical columns
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
        # Random Forest: Ensemble của nhiều decision trees
        # Mỗi tree được train trên subset ngẫu nhiên của dữ liệu và features
        classifier = RandomForestClassifier(
            n_estimators=300,  # Số lượng decision trees trong forest (càng nhiều càng tốt nhưng chậm hơn)
            max_depth=None,  # Không giới hạn độ sâu của tree (để tree phát triển đầy đủ)
            n_jobs=-1,  # Sử dụng tất cả CPU cores để train song song (tăng tốc)
            random_state=42,  # Seed để tái lập kết quả (reproducibility)
            class_weight="balanced_subsample",  # Tự động cân bằng class weights cho mỗi tree (xử lý imbalanced data)
        )
        # Tạo pipeline: preprocess dữ liệu trước, sau đó train classifier
        return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])
    
    # ========== XGBOOST ==========
    elif model_type == "xgboost":
        # Kiểm tra xem XGBoost đã được cài đặt chưa
        if XGBClassifier is None:
            raise ImportError(
                "XGBoost chưa được cài đặt. Chạy: pip install xgboost"
            )
        # XGBoost: Gradient Boosting mạnh mẽ, xử lý missing values tốt
        # Sử dụng gradient descent để tối ưu loss function
        classifier = XGBClassifier(
            n_estimators=300,  # Số lượng boosting rounds (số trees)
            max_depth=6,  # Độ sâu tối đa của mỗi tree (tránh overfitting)
            learning_rate=0.1,  # Tốc độ học (shrinkage) - nhỏ hơn = ổn định hơn nhưng cần nhiều trees hơn
            random_state=42,  # Seed để tái lập
            n_jobs=-1,  # Sử dụng tất cả CPU cores
            eval_metric="mlogloss",  # Metric đánh giá cho multi-class classification
        )
        return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])
    
    # ========== LIGHTGBM ==========
    elif model_type == "lightgbm":
        # Kiểm tra xem LightGBM đã được cài đặt chưa
        if LGBMClassifier is None:
            raise ImportError(
                "LightGBM chưa được cài đặt. Chạy: pip install lightgbm"
            )
        # LightGBM: Gradient Boosting nhanh và hiệu quả, sử dụng leaf-wise growth
        # Tốt hơn XGBoost về tốc độ và memory usage
        classifier = LGBMClassifier(
            n_estimators=300,  # Số lượng boosting rounds
            max_depth=6,  # Độ sâu tối đa
            learning_rate=0.1,  # Tốc độ học
            random_state=42,  # Seed
            n_jobs=-1,  # Sử dụng tất cả CPU cores
            class_weight="balanced",  # Tự động cân bằng class weights
        )
        return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])
    
    # ========== ENSEMBLE VOTING ==========
    elif model_type in ["ensemble", "voting"]:
        # Ensemble Voting: Kết hợp nhiều model khác nhau
        # Mỗi model đưa ra dự đoán, label được vote nhiều nhất sẽ là kết quả cuối cùng
        # Ưu điểm: Giảm overfitting, tăng độ ổn định, kết hợp ưu điểm của nhiều model
        
        # Danh sách các estimators (model) sẽ được kết hợp
        estimators = []
        
        # 1. Random Forest - ổn định, không cần tuning nhiều
        estimators.append(
            (
                "rf",  # Tên identifier cho model này
                RandomForestClassifier(
                    n_estimators=100,  # Giảm số trees để tăng tốc training
                    max_depth=None,  # Không giới hạn độ sâu
                    n_jobs=-1,  # Parallel processing
                    random_state=42,  # Seed
                    class_weight="balanced_subsample",  # Cân bằng classes
                ),
            )
        )
        
        # 2. Extra Trees - tương tự RF nhưng randomness cao hơn
        estimators.append(
            (
                "et",
                ExtraTreesClassifier(
                    n_estimators=100,  # Giảm số trees để tăng tốc training
                    max_depth=None,  # Không giới hạn độ sâu
                    n_jobs=-1,  # Parallel processing
                    random_state=42,  # Seed
                    class_weight="balanced_subsample",  # Cân bằng classes
                ),
            )
        )
        
        # 3. GradientBoostingClassifier (sklearn native) - khác với XGBoost/LightGBM
        estimators.append(
            (
                "gbc",
                GradientBoostingClassifier(
                    n_estimators=80,  # Giảm số boosting rounds để tăng tốc
                    max_depth=5,  # Độ sâu tối đa
                    learning_rate=0.1,  # Tốc độ học
                    random_state=42,  # Seed
                    subsample=0.8,  # Fraction of samples for each tree
                ),
            )
        )
        
        # 4. AdaBoostClassifier - Adaptive Boosting với Decision Tree base
        # Lưu ý: sklearn >= 1.2 dùng 'estimator', < 1.2 dùng 'base_estimator'
        try:
            ada_estimator = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=3,  # Shallow trees cho AdaBoost
                    random_state=42,
                    class_weight="balanced",
                ),
                n_estimators=50,  # Giảm số weak learners để tăng tốc
                learning_rate=0.8,  # Tốc độ học
                random_state=42,
            )
        except TypeError:
            # Fallback cho sklearn < 1.2
            ada_estimator = AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(
                    max_depth=3,
                    random_state=42,
                    class_weight="balanced",
                ),
                n_estimators=50,  # Giảm số weak learners để tăng tốc
                learning_rate=0.8,
                random_state=42,
            )
        estimators.append(("ada", ada_estimator))
        
        # 5. DecisionTreeClassifier với max_depth khác - đơn giản nhưng hiệu quả
        estimators.append(
            (
                "dt_deep",
                DecisionTreeClassifier(
                    max_depth=15,  # Độ sâu lớn hơn
                    min_samples_split=10,  # Minimum samples để split
                    min_samples_leaf=5,  # Minimum samples ở leaf
                    random_state=42,
                    class_weight="balanced",
                ),
            )
        )
        
        # 6. LogisticRegression - Linear model, khác hoàn toàn với tree-based
        estimators.append(
            (
                "lr",
                LogisticRegression(
                    max_iter=500,  # Giảm số iterations để tăng tốc
                    random_state=42,
                    class_weight="balanced",  # Cân bằng classes
                    n_jobs=-1,  # Parallel processing
                    solver="lbfgs",  # Solver cho multi-class
                    multi_class="multinomial",  # Multi-class strategy
                ),
            )
        )
        
        # 7. MLPClassifier - Neural Network, khác với tree-based models
        estimators.append(
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(100, 50),  # 2 hidden layers: 100 và 50 neurons
                    max_iter=300,  # Giảm số iterations để tăng tốc
                    random_state=42,
                    early_stopping=True,  # Early stopping để tránh overfitting
                    validation_fraction=0.1,  # 10% data cho validation
                    n_iter_no_change=10,  # Stop nếu không cải thiện sau 10 iterations
                    learning_rate_init=0.01,  # Learning rate
                    solver="adam",  # Adam optimizer
                ),
            )
        )
        
        # 8. CatBoost - Nếu có, thêm vào (tốt với categorical features)
        if CatBoostClassifier is not None:
            estimators.append(
                (
                    "catboost",
                    CatBoostClassifier(
                        iterations=80,  # Giảm số boosting rounds để tăng tốc
                        depth=6,  # Độ sâu
                        learning_rate=0.1,  # Tốc độ học
                        random_state=42,
                        verbose=False,  # Không log chi tiết
                        class_weights="balanced",  # Cân bằng classes
                    ),
                )
            )
        else:
            logging.warning("CatBoost chưa được cài, bỏ qua trong ensemble")
        
        # 9. BaggingClassifier với DecisionTree - Ensemble của ensembles
        # Lưu ý: sklearn >= 1.2 dùng 'estimator', < 1.2 dùng 'base_estimator'
        try:
            bagging_estimator = BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=10,
                    random_state=42,
                    class_weight="balanced",
                ),
                n_estimators=30,  # Giảm số base estimators để tăng tốc
                random_state=42,
                n_jobs=-1,  # Parallel processing
            )
        except TypeError:
            # Fallback cho sklearn < 1.2
            bagging_estimator = BaggingClassifier(
                base_estimator=DecisionTreeClassifier(
                    max_depth=10,
                    random_state=42,
                    class_weight="balanced",
                ),
                n_estimators=30,  # Giảm số base estimators để tăng tốc
                random_state=42,
                n_jobs=-1,
            )
        estimators.append(("bagging", bagging_estimator))
        
        # 10. Random Forest với cấu hình khác (nhiều trees hơn, depth giới hạn)
        estimators.append(
            (
                "rf_deep",
                RandomForestClassifier(
                    n_estimators=80,  # Giảm số trees để tăng tốc
                    max_depth=20,  # Giới hạn depth
                    min_samples_split=5,  # Minimum samples để split
                    n_jobs=-1,
                    random_state=42,
                    class_weight="balanced_subsample",
                ),
            )
        )
        
        # Kiểm tra có ít nhất 2 models để ensemble
        if len(estimators) < 2:
            raise ValueError(
                "Cần ít nhất 2 models để ensemble. "
                "Hiện tại chỉ có %d models." % len(estimators)
            )
        
        # VotingClassifier: Kết hợp các models bằng voting
        classifier = VotingClassifier(
            estimators=estimators,  # Danh sách các models
            voting="hard",  # 'hard' = majority vote (chọn label được vote nhiều nhất)
            # 'soft' = dùng xác suất, tính trung bình weighted (cần probability=True)
            n_jobs=-1,  # Parallel processing khi predict
        )
        
        logging.info(
            "Sử dụng Ensemble Voting với %d models: %s",
            len(estimators),
            [name for name, _ in estimators],
        )
        return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])
    
    # ========== MODEL TYPE KHÔNG HỢP LỆ ==========
    else:
        raise ValueError(
            f"Model type không hợp lệ: {model_type}. "
            "Chọn một trong: random_forest, xgboost, lightgbm, ensemble, voting"
        )


def evaluate_model(
    model: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series
) -> Dict[str, Dict[str, float]]:
    """
    Đánh giá model trên tập dữ liệu evaluation và trả về metrics.
    
    Args:
        model: Pipeline đã được train (preprocessor + classifier)
        X_eval: Features của tập evaluation
        y_eval: Labels thực tế của tập evaluation
    
    Returns:
        Dictionary chứa:
        - classification_report: Precision, recall, F1-score cho từng class
        - confusion_matrix: Ma trận nhầm lẫn (confusion matrix)
    """
    logging.info("Đang đánh giá mô hình...")
    # Dự đoán labels cho tập evaluation
    y_pred = model.predict(X_eval)
    
    # Tạo classification report (precision, recall, F1 cho từng class)
    # output_dict=True: Trả về dạng dict thay vì string
    # zero_division=0: Nếu chia cho 0 thì trả về 0 (không báo lỗi)
    report = classification_report(y_eval, y_pred, output_dict=True, zero_division=0)
    
    # Tạo confusion matrix: so sánh y thực tế vs y dự đoán
    conf_mtx = confusion_matrix(y_eval, y_pred)
    
    # Log kết quả để xem trong console
    logging.info("Classification report:\n%s", json.dumps(report, indent=2))
    logging.info("Confusion matrix:\n%s", conf_mtx)
    
    # Trả về dạng dict để lưu vào file JSON sau
    return {
        "classification_report": report,  # Metrics chi tiết
        "confusion_matrix": conf_mtx.tolist(),  # Confusion matrix (convert numpy array sang list)
    }


def save_artifacts(
    model: Pipeline,
    metrics: Dict[str, Dict[str, float]],
    output_dir: Path,
    metadata: Dict[str, str | int | float],
) -> None:
    """
    Lưu các artifacts: model, metrics, metadata.
    
    Args:
        model: Pipeline đã được train (sẽ lưu dạng joblib)
        metrics: Dictionary chứa metrics đánh giá (sẽ lưu dạng JSON)
        output_dir: Thư mục lưu artifacts
        metadata: Dictionary chứa thông tin về quá trình training (sẽ lưu dạng JSON)
    """
    # Tạo thư mục output nếu chưa tồn tại
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Định nghĩa đường dẫn các file sẽ lưu
    model_path = output_dir / "ids_pipeline.joblib"  # Model pipeline
    metrics_path = output_dir / "metrics.json"  # Metrics đánh giá
    metadata_path = output_dir / "metadata.json"  # Metadata training

    # Lưu model pipeline dạng joblib (format của sklearn)
    # Joblib nhanh hơn pickle cho numpy arrays lớn
    joblib.dump(model, model_path)
    logging.info("Đã lưu pipeline vào %s", model_path)

    # Thử lưu thêm dạng H5 (nếu có Keras) - optional
    try:
        import keras  # Import lazy để tránh phụ thuộc khi không cần H5
    except ImportError:
        keras = None  # Nếu chưa cài Keras thì bỏ qua

    # Nếu có Keras, thử lưu thêm format H5 (cho tương thích với các hệ thống khác)
    if keras is not None:
        keras_path = output_dir / "ids_pipeline_model_high_level.h5"
        try:
            keras.models.save_model(model, keras_path)
            logging.info("Đã lưu thêm mô hình dạng H5 vào %s", keras_path)
        except Exception as exc:
            # Nếu không lưu được H5 thì chỉ warning, không fail
            logging.warning("Không thể lưu H5: %s", exc)

    # Lưu metrics dạng JSON (dễ đọc và xử lý sau)
    # make_json_safe: Convert numpy types sang Python native types
    metrics_path.write_text(json.dumps(make_json_safe(metrics), indent=2), encoding="utf-8")
    logging.info("Đã lưu metrics vào %s", metrics_path)

    # Lưu metadata dạng JSON
    metadata_path.write_text(json.dumps(make_json_safe(metadata), indent=2), encoding="utf-8")
    logging.info("Đã lưu metadata vào %s", metadata_path)


def main() -> None:
    """
    Hàm main: Điểm vào chính của script.
    
    Đọc arguments từ CLI, setup logging, và gọi training pipeline.
    """
    # Parse arguments từ command line
    args = parse_args()
    # Setup logging để hiển thị thông tin trong quá trình chạy
    setup_logging()

    # Gọi training pipeline với tất cả các tham số từ CLI
    run_training_pipeline(
        splits_dir=args.splits_dir,  # Thư mục chứa train/val/test splits
        source_dataset=args.source_dataset,  # Dataset nguồn để split nếu cần
        auto_split=args.auto_split,  # Tự động split nếu chưa có
        split_script=args.split_script,  # Đường dẫn script split
        train_variant=args.train_variant,  # Dùng train_raw hay train_balanced
        label_column=args.label_column,  # Tên cột label
        drop_columns=args.drop_columns,  # Các cột cần bỏ qua
        test_size=args.test_size,  # Tỷ lệ test nếu muốn tách lại train
        random_state=args.random_state,  # Seed để tái lập
        sample_frac=args.sample_frac,  # Tỷ lệ sample dữ liệu (để test nhanh)
        output_dir=args.output_dir,  # Thư mục lưu artifacts
        model_type=args.model_type,  # Loại model sử dụng (random_forest, xgboost, ensemble, etc.)
    )


def run_training_pipeline(
    *,
    splits_dir: Path | str,  # Thư mục chứa train/val/test splits
    source_dataset: Path | str,  # Dataset nguồn để split nếu cần
    auto_split: bool,  # Tự động split nếu chưa có
    split_script: Path | str,  # Đường dẫn script split
    train_variant: str = "balanced",  # Dùng train_raw hay train_balanced
    label_column: str = "label_group_encoded",  # Tên cột label
    drop_columns: List[str] | None = None,  # Các cột cần bỏ qua
    test_size: float | None = None,  # Tỷ lệ test nếu muốn tách lại train
    random_state: int = 42,  # Seed để tái lập
    sample_frac: float | None = None,  # Tỷ lệ sample dữ liệu
    output_dir: Path | str = Path("artifacts"),  # Thư mục lưu artifacts
    model_type: str = "ensemble",  # Loại model: random_forest, xgboost, lightgbm, ensemble
) -> Dict[str, object]:
    """
    Chạy toàn bộ quy trình huấn luyện level 1.
    
    Quy trình:
    1. Chuẩn hóa đường dẫn
    2. Kiểm tra và tự động split dữ liệu nếu cần
    3. Đọc train/val/test datasets
    4. Tách features và labels
    5. Xây dựng và train model
    6. Đánh giá trên validation và test
    7. Lưu artifacts (model, metrics, metadata)
    
    Returns:
        Dictionary chứa pipeline, metrics, metadata, output_dir
    """
    setup_logging()  # Setup logging để hiển thị thông tin

    # ========== CHUẨN HÓA ĐƯỜNG DẪN ==========
    # Lấy thư mục hiện tại của file này (ids_pipeline/)
    script_dir = Path(__file__).resolve().parent
    # Lấy thư mục gốc của dự án (parent của ids_pipeline/)
    project_root = script_dir.parent

    # Resolve đường dẫn splits_dir: nếu là relative thì tính từ project root
    resolved_splits_dir = Path(splits_dir)
    if not resolved_splits_dir.is_absolute():  # Nếu là đường dẫn tương đối
        resolved_splits_dir = (project_root / resolved_splits_dir).resolve()

    # Resolve đường dẫn source_dataset
    resolved_source_dataset = Path(source_dataset)
    if not resolved_source_dataset.is_absolute():
        resolved_source_dataset = (project_root / resolved_source_dataset).resolve()

    # Resolve đường dẫn split_script
    resolved_split_script = Path(split_script)
    if not resolved_split_script.is_absolute():
        resolved_split_script = (project_root / resolved_split_script).resolve()

    # Resolve đường dẫn output_dir
    resolved_output = Path(output_dir)
    if not resolved_output.is_absolute():
        resolved_output = (project_root / resolved_output).resolve()
    
    # Bảo vệ trường hợp drop_columns là None (set thành empty list)
    effective_drop = drop_columns or []

    # ========== CHỌN FILE TRAIN ==========
    # Chọn file train tương ứng với biến thể yêu cầu (raw hoặc balanced)
    if train_variant == "raw":
        # train_raw: Dữ liệu gốc chưa được cân bằng (có thể mất cân bằng classes)
        train_file = resolved_splits_dir / "train_raw.pkl"
    elif train_variant == "balanced":
        # train_balanced: Dữ liệu đã được cân bằng (oversample/undersample)
        train_file = resolved_splits_dir / "train_balanced.pkl"
    else:
        raise ValueError(f"Loại tập dữ liệu không hợp lệ: {train_variant}")

    # ========== KIỂM TRA VÀ TỰ ĐỘNG SPLIT ==========
    # Danh sách các file cần thiết để training
    required_files = [
        train_file,  # File train (raw hoặc balanced)
        resolved_splits_dir / "val.pkl",  # File validation
        resolved_splits_dir / "test.pkl",  # File test
    ]
    
    # Nếu bật auto_split và thiếu file, tự động gọi split_dataset.py để tạo mới
    if auto_split and not all(path.exists() for path in required_files):
        logging.info(
            "Không thấy đủ file split tại %s, gọi split_dataset.py level 1...",
            resolved_splits_dir,
        )
        # Chuẩn bị lệnh gọi script split_dataset.py để tự động split dữ liệu
        cmd = [
            sys.executable,  # Python executable hiện tại
            str(resolved_split_script),  # Đường dẫn script split_dataset.py
            "--source",
            str(resolved_source_dataset),  # Dataset nguồn
            "--level",
            "1",  # Level 1: Split theo label_group
            "--label-column",
            str(label_column),  # Cột label để split
            "--output-dir",
            str(resolved_splits_dir),  # Thư mục lưu splits
            "--train-min",
            str(10_000),  # Số mẫu tối thiểu mỗi class trong train (oversample nếu thiếu)
            "--train-max",
            str(200_000),  # Số mẫu tối đa mỗi class trong train (undersample nếu dư)
            "--random-state",
            str(random_state),  # Seed để tái lập
        ]
        logging.info("Chạy lệnh: %s", " ".join(cmd))
        # Chạy lệnh và kiểm tra lỗi (check=True: raise exception nếu fail)
        subprocess.run(cmd, check=True)

    # ========== ĐỌC DỮ LIỆU ==========
    # Đọc các tập dữ liệu (train/val/test) vừa tạo hoặc có sẵn
    df_train = load_split_dataframe(train_file, sample_frac, random_state)
    df_val = load_split_dataframe(
        resolved_splits_dir / "val.pkl", None, random_state
    )  # Validation set (không sample)
    df_test = load_split_dataframe(
        resolved_splits_dir / "test.pkl", None, random_state
    )  # Test set (không sample)

    # ========== TÁCH LẠI TRAIN (TÙY CHỌN) ==========
    # Có thể tách lại train thành 2 phần (train + holdout) nếu test_size được thiết lập
    # Điều này hữu ích khi muốn có thêm một tập test riêng để đánh giá
    df_train_for_model = df_train  # Dữ liệu dùng để train
    df_holdout = df_test  # Dữ liệu dùng để đánh giá cuối cùng (holdout test)
    
    # Nếu có test_size, tách lại train thành train + holdout
    if test_size is not None and 0 < test_size < 1:
        # Tách train với stratify để giữ tỷ lệ classes
        df_train_for_model, df_holdout = train_test_split(
            df_train,  # Dữ liệu train gốc
            test_size=test_size,  # Tỷ lệ holdout
            stratify=df_train[label_column],  # Giữ tỷ lệ classes khi split
            random_state=random_state,  # Seed để tái lập
        )
        logging.info(
            "Đã tách lại train thành train/test với test_size=%.2f -> train=%d, test=%d",
            test_size,
            df_train_for_model.shape[0],
            df_holdout.shape[0],
        )

    # ========== TÁCH FEATURES VÀ LABELS ==========
    # Tách nhãn và đặc trưng cho từng tập (X = features, y = labels)
    X_train, y_train, label_actual, drop_cols_resolved = prepare_features_labels(
        df_train_for_model, label_column, effective_drop
    )
    X_val, y_val, _, _ = prepare_features_labels(df_val, label_column, effective_drop)
    X_holdout, y_holdout, _, _ = prepare_features_labels(
        df_holdout, label_column, effective_drop
    )
    logging.info("Sử dụng cột nhãn: %s", label_actual)

    # Xây dựng pipeline tiền xử lý và mô hình.
    # Preprocessor: Xử lý missing values, scaling, encoding
    preprocessor = build_preprocess_transformer(X_train)
    # Pipeline: Kết hợp preprocessor + classifier (theo model_type)
    pipeline = build_model_pipeline(preprocessor, model_type=model_type)

    # Huấn luyện mô hình trên tập train chuẩn hóa.
    logging.info(
        "Bắt đầu huấn luyện %s (train=%d)...", model_type.upper(), X_train.shape[0]
    )
    # Fit pipeline: Preprocess dữ liệu rồi train classifier
    pipeline.fit(X_train, y_train)
    logging.info("Huấn luyện hoàn tất.")

    # ========== ĐÁNH GIÁ MODEL ==========
    # Đánh giá trên tập validation (dùng để chọn hyperparameters, early stopping)
    metrics_val = evaluate_model(pipeline, X_val, y_val)
    # Đánh giá trên tập holdout/test (đánh giá cuối cùng, không được dùng để tune)
    metrics_holdout = evaluate_model(pipeline, X_holdout, y_holdout)

    # ========== LƯU METADATA ==========
    # Ghi lại thông tin phục vụ tái hiện thí nghiệm (reproducibility)
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
        "model_type": model_type,  # Lưu loại model đã sử dụng
        "class_labels": sorted(y_train.unique()),  # Danh sách các class labels
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


