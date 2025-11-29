"""
Script tiền xử lý dataset: đọc dữ liệu đã load (pickle hoặc CSV), làm sạch và lưu lại pickle.

Ví dụ chạy:
python scripts/preprocess_dataset.py --source dataset.pkl --output dataset_clean.pkl
"""

from __future__ import annotations
# Cho phép dùng cú pháp type annotation mới mà vẫn chạy trên phiên bản Python cũ hơn.

import argparse  # Đọc tham số từ dòng lệnh.
import json  # Xuất metadata tiền xử lý để tái sử dụng ở các bước sau.
import re  # Xử lý chuỗi với biểu thức chính quy.
from pathlib import Path  # Làm việc với đường dẫn dạng object.
from typing import Iterable  # Type hint cho các tham số dạng lặp.

import numpy as np  # Hỗ trợ thao tác số học, ví dụ phát hiện giá trị vô hạn.
import pandas as pd  # Thư viện xử lý dữ liệu dạng bảng.

BASE_DIR = Path(__file__).resolve().parent.parent  # Thư mục gốc của dự án.


def parse_args() -> argparse.Namespace:
    """Đọc cấu hình dòng lệnh cho bước tiền xử lý."""
    parser = argparse.ArgumentParser(
        description=(
            "Tiền xử lý dataset (chuẩn hóa tên cột, xử lý thiếu, mã hóa nhãn, làm giàu đặc trưng) "
            "và lưu pickle."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset.pkl"),
        help="Đường dẫn dữ liệu đầu vào (pickle hoặc CSV). Mặc định: dataset.pkl.",
    )
    parser.add_argument(
        "--fallback-csv",
        type=Path,
        default=Path("dataset.csv"),
        help="CSV dùng để fallback nếu pickle chưa tồn tại (mặc định: dataset.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_clean.pkl"),
        help="Đường dẫn lưu dữ liệu đã tiền xử lý dạng pickle (mặc định: dataset_clean.pkl).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="(Tuỳ chọn) Lưu thêm bản CSV sạch.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="Label",
        help="Tên cột nhãn (phân biệt hoa thường, mặc định: Label).",
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Loại bỏ dòng trùng lặp sau khi làm sạch.",
    )
    parser.add_argument(
        "--min-non-null-ratio",
        type=float,
        default=0.5,
        help="Ngưỡng tối thiểu (0-1) tỷ lệ giá trị không null để giữ cột (mặc định 0.5).",
    )
    parser.add_argument(
        "--drop-constant-columns",
        action="store_true",
        help="Loại bỏ các cột chỉ có một giá trị duy nhất.",
    )
    parser.add_argument(
        "--outlier-method",
        choices=("none", "iqr_clip"),
        default="none",
        help="Phương pháp xử lý ngoại lệ: none (mặc định) hoặc iqr_clip (clip theo IQR).",
    )
    parser.add_argument(
        "--iqr-factor",
        type=float,
        default=1.5,
        help="Hệ số nhân IQR để clip ngoại lệ (mặc định 1.5).",
    )
    parser.add_argument(
        "--scale-method",
        choices=("none", "standard", "minmax"),
        default="none",
        help=(
            "Chuẩn hóa dữ liệu số: none (mặc định), standard (z-score) hoặc minmax (0-1). "
            "⚠️ CẢNH BÁO: Nên để 'none' vì model training pipeline đã có StandardScaler. "
            "Nếu scale ở đây sẽ bị double scaling → kết quả prediction sai!"
        ),
    )
    parser.add_argument(
        "--one-hot",
        action="store_true",
        help="Bật one-hot encoding cho các cột phân loại (ngoại trừ cột nhãn).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="In thống kê tổng quan sau tiền xử lý.",
    )
    parser.add_argument(
        "--create-label-group",
        action="store_true",
        help="Sinh thêm cột label_group (gom nhóm nhãn chính).",
    )
    parser.add_argument(
        "--label-group-column",
        type=str,
        default="label_group",
        help="Tên cột chứa nhãn nhóm (mặc định: label_group).",
    )
    parser.set_defaults(create_label_group=True)
    parser.add_argument(  # Tham số lựa chọn phương pháp cân bằng.
        "--balance-method",
        choices=("none", "oversample", "undersample"),  # Các phương án hỗ trợ.
        default="none",  # Mặc định không cân bằng.
        help="Cân bằng dữ liệu theo nhãn: none (giữ nguyên), oversample (nhân bản lớp hiếm), undersample (cắt bớt lớp lớn).",
    )
    parser.add_argument(  # Seed giúp tái lập kết quả cân bằng.
        "--balance-random-state",
        type=int,
        default=42,
        help="Seed ngẫu nhiên cho bước cân bằng dữ liệu (mặc định: 42).",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="(Tuỳ chọn) Lưu metadata tiền xử lý (JSON).",
    )
    return parser.parse_args()


def normalize_column(name: str) -> str:
    """Chuẩn hóa tên cột về dạng snake_case chữ thường, bỏ ký tự đặc biệt."""
    cleaned = name.strip()  # Loại bỏ khoảng trắng ở hai đầu.
    cleaned = re.sub(r"[^\w]+", "_", cleaned, flags=re.UNICODE)  # Thay ký tự đặc biệt bằng dấu gạch dưới.
    cleaned = re.sub(r"_+", "_", cleaned)  # Gom các dấu '_' liên tiếp về một dấu.
    return cleaned.strip("_").lower()  # Xóa '_' dư thừa ở biên và đổi về chữ thường.


def load_raw_dataframe(source: Path, fallback_csv: Path | None) -> pd.DataFrame:
    """Load dataframe từ pickle/CSV; nếu nguồn chính không có sẽ fallback sang CSV."""
    if source.exists():
        if source.suffix.lower() == ".pkl":
            print(f"Đang đọc pickle: {source}")
            return pd.read_pickle(source)  # Đọc nhanh hơn và giữ nguyên kiểu dữ liệu.
        if source.suffix.lower() == ".csv":
            print(f"Đang đọc CSV: {source}")
            return pd.read_csv(source, low_memory=False)  # Đọc CSV với dtype ổn định hơn.
        raise ValueError(f"Định dạng nguồn không hỗ trợ: {source.suffix}")

    if fallback_csv and fallback_csv.exists():
        print(f"Không thấy {source}, fallback sang CSV: {fallback_csv}")
        return pd.read_csv(fallback_csv, low_memory=False)  # Đọc file dự phòng nếu có.

    raise FileNotFoundError(f"Không tìm thấy dữ liệu đầu vào ở {source} hoặc {fallback_csv}.")


def convert_numeric(df: pd.DataFrame, skip_cols: Iterable[str]) -> pd.DataFrame:
    """Cố gắng ép các cột sang số (trừ những cột bị loại trừ)."""
    for col in df.columns:
        if col in skip_cols:
            continue  # Không động vào các cột cần giữ nguyên (ví dụ cột nhãn).
        if pd.api.types.is_numeric_dtype(df[col]):
            continue  # Nếu đã là số thì không cần xử lý thêm.
        coerced = pd.to_numeric(df[col], errors="coerce")  # Thử chuyển về kiểu số; lỗi sẽ thành NaN.
        df[col] = coerced  # Gán lại cột đã được xử lý.
    return df


def fill_missing_values(df: pd.DataFrame, skip_cols: Iterable[str]) -> pd.DataFrame:
    """Điền giá trị thiếu: median cho cột số, mode cho cột phân loại."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.difference(skip_cols)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    # Median ổn định hơn mean khi có ngoại lệ, phù hợp để điền thiếu cho dữ liệu số.

    non_numeric_cols = df.columns.difference(numeric_cols.union(skip_cols))
    for col in non_numeric_cols:
        if df[col].isna().any():
            mode = df[col].mode(dropna=True)  # Mode đại diện cho giá trị phổ biến nhất.
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])  # Điền thiếu bằng mode đầu tiên.
            else:
                df[col] = df[col].fillna("")  # Nếu không xác định được mode, đặt rỗng để nhất quán.
    return df


def encode_label(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, dict[int, str]]:
    """Mã hóa cột nhãn sang số và trả về mapping ngược."""
    if label_col not in df.columns:
        print(f"Không tìm thấy cột nhãn {label_col}, bỏ qua bước mã hóa.")
        return df, {}

    label_series = df[label_col].astype(str).str.strip()  # Đồng nhất kiểu dữ liệu và bỏ khoảng trắng.
    df[label_col] = label_series  # Cập nhật lại cột gốc sau khi làm sạch.
    codes, uniques = pd.factorize(label_series, sort=True)  # Mã hóa nhãn thành số nguyên.
    df[f"{label_col}_encoded"] = codes  # Thêm cột nhãn đã mã hóa.
    mapping = {code: label for code, label in enumerate(uniques)}  # Bảng tra cứu để giải mã lại.
    return df, mapping


def add_label_group_column(
    df: pd.DataFrame,
    source_col: str,
    group_col: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Tạo cột gom nhóm nhãn theo logic tùy chỉnh."""
    if source_col not in df.columns:
        print(f"Không tìm thấy cột nhãn {source_col} để tạo nhóm.")
        return df, {}

    # Định nghĩa mapping từ nhãn chi tiết sang nhóm tổng quát.
    # Chỉ giữ lại: benign, dos, ddos, portscan
    # Đã bỏ: Bot, Infiltration, Heartbleed
    group_rules = {
        "benign": "benign",
        "dos hulk": "dos",
        "dos goldeneye": "dos",
        "dos slowloris": "dos",
        "dos slowhttptest": "dos",
        "ddos": "ddos",
        "portscan": "portscan",
        # Bot, Infiltration, Heartbleed đã được loại bỏ ở bước trước
    }

    default_group = "other"  # Phân loại mặc định cho nhãn chưa định nghĩa.
    mapping_report: dict[str, str] = {}

    cleaned_series = df[source_col].astype(str).str.strip()
    # Xây dựng mapping thực tế từ nhãn gốc -> nhóm.
    for original_label in cleaned_series.unique():
        normalized = original_label.lower()
        group_name = group_rules.get(normalized, default_group)
        mapping_report[original_label] = group_name

    group_series = cleaned_series.map(mapping_report)
    df[group_col] = group_series  # Gán vào DataFrame.
    return df, mapping_report


def add_binary_label_column(
    df: pd.DataFrame,
    group_col: str,
    binary_col: str = "label_binary_encoded"
) -> pd.DataFrame:
    """Tạo cột binary label: 0=benign, 1=attack (gộp dos, ddos, portscan)"""
    if group_col not in df.columns:
        print(f"Không tìm thấy cột nhóm {group_col} để tạo binary label.")
        return df
    
    if binary_col in df.columns:
        print(f"Cột {binary_col} đã tồn tại, bỏ qua.")
        return df
    
    def map_to_binary(group_value):
        if pd.isna(group_value):
            return 0
        group_str = str(group_value).lower().strip()
        if group_str == "benign":
            return 0
        else:  # dos, ddos, portscan, other -> attack
            return 1
    
    df[binary_col] = df[group_col].apply(map_to_binary)
    print(f"Đã tạo cột binary label: {binary_col}")
    print(f"Binary distribution: {df[binary_col].value_counts().to_dict()}")
    return df


def add_attack_type_label_column(
    df: pd.DataFrame,
    group_col: str,
    attack_type_col: str = "label_attack_type_encoded"
) -> pd.DataFrame:
    """Tạo cột attack type label: 0=dos, 1=ddos, 2=portscan (chỉ cho attack, benign = -1)"""
    if group_col not in df.columns:
        print(f"Không tìm thấy cột nhóm {group_col} để tạo attack type label.")
        return df
    
    if attack_type_col in df.columns:
        print(f"Cột {attack_type_col} đã tồn tại, bỏ qua.")
        return df
    
    def map_to_attack_type(group_value):
        if pd.isna(group_value):
            return -1
        group_str = str(group_value).lower().strip()
        if group_str == "benign":
            return -1  # Không phải attack
        elif group_str == "dos":
            return 0
        elif group_str == "ddos":
            return 1
        elif group_str == "portscan":
            return 2
        else:
            return -1  # Unknown
    
    df[attack_type_col] = df[group_col].apply(map_to_attack_type)
    print(f"Đã tạo cột attack type label: {attack_type_col}")
    print(f"Attack type distribution: {df[attack_type_col].value_counts().to_dict()}")
    return df


def drop_sparse_columns(
    df: pd.DataFrame, min_ratio: float, skip_cols: Iterable[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Loại bỏ các cột có tỷ lệ giá trị hợp lệ thấp hơn ngưỡng cho phép."""
    if min_ratio <= 0:
        return df, []
    dropped: list[str] = []
    for col in df.columns:
        if col in skip_cols:
            continue
        non_null_ratio = df[col].notna().mean()  # Tỷ lệ phần tử không rỗng trên toàn bộ cột.
        if not np.isfinite(non_null_ratio) or non_null_ratio < min_ratio:
            dropped.append(col)
    if dropped:
        df = df.drop(columns=dropped)
    return df, dropped


def drop_constant_columns(
    df: pd.DataFrame, skip_cols: Iterable[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Loại bỏ các cột chỉ có một giá trị (bao gồm cả NaN)."""
    dropped: list[str] = []
    for col in df.columns:
        if col in skip_cols:
            continue
        if df[col].nunique(dropna=False) <= 1:  # Cột có 0-1 giá trị => không hữu dụng.
            dropped.append(col)
    if dropped:
        df = df.drop(columns=dropped)
    return df, dropped


def clip_outliers_iqr(
    df: pd.DataFrame, numeric_cols: Iterable[str], factor: float
) -> dict[str, dict[str, float]]:
    """Clip ngoại lệ dựa trên khoảng tứ phân vị (IQR)."""
    if factor <= 0:
        return {}
    clip_info: dict[str, dict[str, float]] = {}
    for col in numeric_cols:
        series = df[col]
        if not np.issubdtype(series.dtype, np.number):
            continue
        q1 = series.quantile(0.25)  # Phân vị thứ 25.
        q3 = series.quantile(0.75)  # Phân vị thứ 75.
        iqr = q3 - q1  # Khoảng tứ phân vị.
        if not np.isfinite(iqr) or iqr == 0:
            continue
        lower = q1 - factor * iqr  # Ngưỡng dưới cho clip.
        upper = q3 + factor * iqr  # Ngưỡng trên cho clip.
        df[col] = series.clip(lower=lower, upper=upper)  # Giới hạn giá trị ngoài khoảng.
        clip_info[col] = {"lower": float(lower), "upper": float(upper)}
    return clip_info


def one_hot_encode(
    df: pd.DataFrame, categorical_cols: Iterable[str]
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """One-hot encoding cho các cột phân loại, trả về mapping danh sách giá trị."""
    categorical_cols = list(categorical_cols)
    if not categorical_cols:
        return df, {}
    category_map: dict[str, list[str]] = {}
    for col in categorical_cols:
        uniques = (
            df[col]
            .astype(str)
            .fillna("__nan__")
            .replace("__nan__", np.nan)
            .dropna()
            .unique()
        )
        # Lưu danh sách giá trị để sau này decode/log lại.
        category_map[col] = sorted(str(val) for val in uniques)
    # pd.get_dummies tạo các cột nhị phân tương ứng từng giá trị.
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dummy_na=False)
    return df, category_map


def balance_dataset(
    df: pd.DataFrame,  # Bảng dữ liệu đã làm sạch.
    label_col: str,  # Tên cột nhãn sử dụng để cân bằng.
    method: str,  # Chiến lược cân bằng được chọn.
    random_state: int,  # Seed tạo ngẫu nhiên để tái lập kết quả.
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Cân bằng dữ liệu theo nhãn bằng cách oversample/undersample."""
    if method == "none":  # Nếu người dùng không yêu cầu cân bằng.
        return df, {}  # Trả về dữ liệu gốc và metadata rỗng.
    if label_col not in df.columns:  # Bảo vệ khi cột nhãn không tồn tại.
        print(f"Không tìm thấy cột nhãn {label_col} để cân bằng, bỏ qua.")  # Thông báo cho người dùng.
        return df, {}  # Giữ nguyên dữ liệu.
    counts_before = df[label_col].value_counts().to_dict()  # Ghi nhận phân bố ban đầu để báo cáo.
    rng = np.random.default_rng(random_state)  # Tạo generator ngẫu nhiên dùng chung trong bước oversample.
    target_size = (  # Quy định kích thước mục tiêu cho mỗi lớp.
        max(counts_before.values()) if method == "oversample" else min(counts_before.values())
    )
    balanced_frames: list[pd.DataFrame] = []  # Danh sách chứa từng phần dữ liệu sau xử lý theo lớp.
    for label_value, count in counts_before.items():  # Lặp qua từng nhãn và số lượng hiện tại.
        subset = df[df[label_col] == label_value]  # Chắt lọc dữ liệu thuộc nhãn hiện tại.
        if method == "oversample" and count < target_size:  # Trường hợp lớp nhỏ cần nhân bản.
            need = target_size - count  # Số mẫu cần bổ sung để đạt mức mục tiêu.
            extra_index = rng.choice(  # Chọn ngẫu nhiên chỉ số (có lặp) từ lớp hiện tại.
                subset.index.to_numpy(), size=need, replace=True
            )
            subset = pd.concat([subset, df.loc[extra_index]], ignore_index=False)  # Ghép thêm các bản sao vào lớp.
        if method == "undersample" and count > target_size:  # Trường hợp lớp lớn cần cắt giảm.
            subset = subset.sample(  # Lấy ngẫu nhiên target_size mẫu mà không thay thế.
                n=target_size, random_state=random_state, replace=False
            )
        balanced_frames.append(subset)  # Lưu phần dữ liệu đã điều chỉnh vào danh sách.
    balanced_df = pd.concat(balanced_frames, ignore_index=True)  # Gộp tất cả lớp lại thành DataFrame mới.
    balanced_df = balanced_df.sample(  # Xáo trộn tổng thể để tránh nhóm lớp theo block.
        frac=1, random_state=random_state
    ).reset_index(drop=True)
    counts_after = balanced_df[label_col].value_counts().to_dict()  # Thống kê phân bố sau cân bằng.
    print(f"Đã cân bằng dữ liệu bằng phương pháp {method}.")  # Thông báo phương pháp sử dụng.
    print(f"Phân bố trước: {counts_before}")  # Hiển thị phân bố ban đầu.
    print(f"Phân bố sau  : {counts_after}")  # Hiển thị phân bố sau khi cân bằng.
    return balanced_df, {"before": counts_before, "after": counts_after}  # Trả về dữ liệu mới và metadata.


def scale_numeric_features(
    df: pd.DataFrame, numeric_cols: Iterable[str], method: str
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Chuẩn hóa dữ liệu số theo phương pháp lựa chọn."""
    stats: dict[str, dict[str, float]] = {}
    if method == "none":
        return df, stats
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    for col in numeric_cols:
        series = df[col]
        if not np.issubdtype(series.dtype, np.number):
            continue
        if method == "standard":
            mean = float(series.mean())  # Giá trị trung bình để chuẩn hóa z-score.
            std = float(series.std(ddof=0))  # Độ lệch chuẩn (population).
            if std == 0 or not np.isfinite(std):
                continue
            df[col] = (series - mean) / std  # (x - mean) / std.
            stats[col] = {"mean": mean, "std": std}
        elif method == "minmax":
            min_val = float(series.min())  # Giá trị nhỏ nhất dùng để scale.
            max_val = float(series.max())  # Giá trị lớn nhất dùng để scale.
            if max_val == min_val or not np.isfinite(max_val) or not np.isfinite(min_val):
                continue
            df[col] = (series - min_val) / (max_val - min_val)  # Đưa về khoảng [0, 1].
            stats[col] = {"min": min_val, "max": max_val}
        else:
            raise ValueError(f"Phương pháp scale không hỗ trợ: {method}")
    return df, stats


def print_summary(df: pd.DataFrame, label_col: str) -> None:
    """In thống kê tổng quan sau tiền xử lý."""
    print("\n===== Thống kê dữ liệu sau tiền xử lý =====")
    print(f"Kích thước: {df.shape[0]} dòng x {df.shape[1]} cột")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    print(f"Số cột số: {len(numeric_cols)} | Số cột phân loại: {len(categorical_cols)}")
    if label_col in df.columns:
        print("Phân bố nhãn:")
        print(df[label_col].value_counts(dropna=False).to_string())
    print("===========================================\n")


def resolve_path(path: Path | None) -> Path | None:
    """Chuyển đường dẫn tương đối (so với project root) sang tuyệt đối."""
    if path is None:
        return None
    expanded = path.expanduser()  # Thay thế ký tự '~' nếu có.
    if expanded.is_absolute():
        return expanded  # Nếu đã là đường dẫn tuyệt đối thì dùng luôn.
    return BASE_DIR / expanded


def main() -> None:
    args = parse_args()

    source_path = resolve_path(args.source)  # Đường dẫn dữ liệu đầu vào đã chuẩn hóa.
    fallback_csv = resolve_path(args.fallback_csv)
    output_path = resolve_path(args.output)  # Đường dẫn lưu pickle sau khi xử lý.
    output_csv = resolve_path(args.output_csv) if args.output_csv else None
    metadata_output = resolve_path(args.metadata_output) if args.metadata_output else None

    # 1. Đọc dữ liệu thô từ pickle (ưu tiên) hoặc CSV nếu pickle chưa sẵn.
    df = load_raw_dataframe(source_path, fallback_csv)
    print(f"Dataset gốc: {df.shape[0]} rows x {df.shape[1]} columns")

    # 1.1. Loại bỏ các nhãn không cần thiết: Bot, Infiltration, Heartbleed
    labels_to_remove = ['Bot', 'Infiltration', 'Heartbleed', 'bot', 'infiltration', 'heartbleed']
    before_remove = len(df)
    
    # Tìm cột label thực tế (có thể có khoảng trắng hoặc tên khác)
    actual_label_col = None
    label_col_normalized = args.label_column.strip().lower()
    for col in df.columns:
        if col.strip().lower() == label_col_normalized or 'label' in col.lower():
            actual_label_col = col
            break
    
    if actual_label_col:
        df = df[~df[actual_label_col].astype(str).str.strip().str.lower().isin([l.lower() for l in labels_to_remove])]
        removed_count = before_remove - len(df)
        if removed_count > 0:
            print(f"Đã loại bỏ {removed_count} mẫu với nhãn: {', '.join(labels_to_remove)}")
            print(f"Dataset sau khi loại bỏ: {df.shape[0]} rows x {df.shape[1]} columns")

    # 2. Chuẩn hóa tên cột để dễ thao tác ở các bước sau.
    # Chuẩn hóa toàn bộ tên cột để đảm bảo các bước xử lý sau không bị lỗi vì ký tự lạ.
    df.columns = [normalize_column(col) for col in df.columns]
    # Tên cột nhãn cũng phải chuẩn hóa giống dữ liệu để có thể tra cứu chính xác.
    label_col = normalize_column(args.label_column)

    # 3. Thay thế vô hạn bằng NaN để có thể xử lý thiếu nhất quán.
    df = df.replace([np.inf, -np.inf], np.nan)  # NaN hóa giá trị vô hạn để xử lý thiếu thống nhất.
    # 4. Ép các cột có thể sang kiểu số (trừ cột nhãn).
    df = convert_numeric(df, skip_cols={label_col})
    # 5. Điền giá trị thiếu bằng thống kê phù hợp.
    df = fill_missing_values(df, skip_cols={label_col})

    # 5.1. Loại bỏ dòng thiếu nhãn để tránh lỗi khi mã hóa/cân bằng.
    missing_labels = df[label_col].isna().sum()
    if missing_labels:
        df = df[df[label_col].notna()].reset_index(drop=True)
        print(f"Đã loại bỏ {missing_labels} dòng thiếu nhãn ({label_col}).")

    # 6. Mã hóa nhãn (nếu tồn tại) và chuẩn bị mapping để báo cáo.
    df, label_mapping = encode_label(df, label_col)

    label_group_col: str | None = None  # Tên cột nhóm nhãn nếu được tạo.
    label_group_mapping: dict[str, str] = {}  # Mapping nhãn chi tiết -> nhóm.
    label_group_encoded_mapping: dict[int, str] = {}  # Mapping mã nhóm -> tên nhóm.
    if args.create_label_group:
        # Chuẩn hóa tên cột nhóm để đồng bộ với các cột khác.
        label_group_col = normalize_column(args.label_group_column)
        df, label_group_mapping = add_label_group_column(df, label_col, label_group_col)
        df, label_group_encoded_mapping = encode_label(df, label_group_col)
        
        # Tạo binary label và attack type label cho Level 1 và Level 2
        # Không cần encode vì chúng đã là số (0/1 hoặc 0/1/2/-1)
        binary_col = normalize_column("label_binary_encoded")
        attack_type_col = normalize_column("label_attack_type_encoded")
        df = add_binary_label_column(df, label_group_col, binary_col)
        df = add_attack_type_label_column(df, label_group_col, attack_type_col)
        # Không encode binary và attack_type vì chúng đã là số rồi

    # Tập các cột cần bỏ qua (giữ nguyên) ở những bước xử lý khác.
    skip_cols = {label_col, f"{label_col}_encoded"}
    if label_group_col:
        skip_cols.update({label_group_col, f"{label_group_col}_encoded"})
        # Thêm binary và attack type columns vào skip_cols (không có _encoded vì chúng đã là số)
        binary_col = normalize_column("label_binary_encoded")
        attack_type_col = normalize_column("label_attack_type_encoded")
        if binary_col in df.columns:
            skip_cols.add(binary_col)
        if attack_type_col in df.columns:
            skip_cols.add(attack_type_col)

    # 7. Loại bỏ các cột có quá nhiều giá trị thiếu (quality control).
    df, sparse_dropped = drop_sparse_columns(df, args.min_non_null_ratio, skip_cols)
    if sparse_dropped:
        print(f"Đã loại bỏ {len(sparse_dropped)} cột thiếu dữ liệu: {sparse_dropped}")

    # 8. Loại bỏ các cột không mang thông tin (chỉ có một giá trị).
    constant_dropped: list[str] = []
    if args.drop_constant_columns:
        df, constant_dropped = drop_constant_columns(df, skip_cols)
        if constant_dropped:
            print(f"Đã loại bỏ {len(constant_dropped)} cột constant: {constant_dropped}")

    # 9. Xử lý dữ liệu trùng lặp để tránh bias khi training.
    duplicate_count = 0
    if args.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        duplicate_count = before - len(df)
        if duplicate_count:
            print(f"Đã loại bỏ {duplicate_count} dòng trùng lặp.")

    # 10. Chuẩn hóa ngoại lệ (outlier) bằng cách clip theo khoảng IQR.
    clip_stats: dict[str, dict[str, float]] = {}
    numeric_cols = df.select_dtypes(include=["number"]).columns.difference(skip_cols)
    if args.outlier_method == "iqr_clip":
        clip_stats = clip_outliers_iqr(df, numeric_cols, args.iqr_factor)
        if clip_stats:
            print(f"Đã clip ngoại lệ cho {len(clip_stats)} cột theo IQR (factor={args.iqr_factor}).")

    # 11. Mã hóa one-hot để chuyển biến phân loại sang dạng số nhị phân.
    category_mapping: dict[str, list[str]] = {}
    if args.one_hot:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.difference(skip_cols)
        df, category_mapping = one_hot_encode(df, categorical_cols)
        if category_mapping:
            print(f"Đã one-hot {len(category_mapping)} cột phân loại.")

    # 12. Cân bằng dữ liệu nếu được yêu cầu (oversample/undersample).
    balance_stats: dict[str, dict[str, int]] = {}  # Lưu metadata về cân bằng dữ liệu.
    if args.balance_method != "none":  # Chỉ chạy khi người dùng bật tùy chọn.
        target_col = f"{label_col}_encoded" if f"{label_col}_encoded" in df.columns else label_col  # Ưu tiên dùng nhãn đã mã hóa.
        df, balance_stats = balance_dataset(  # Thực hiện cân bằng theo cấu hình CLI.
            df,
            target_col,
            args.balance_method,
            args.balance_random_state,
        )

    # 12. Chuẩn hóa giá trị số về cùng thang đo (Standard/MinMax) để mô hình dễ học.
    # ⚠️ LƯU Ý QUAN TRỌNG: Model training pipeline (train_level1_rf.py, train_level2_rf.py)
    # đã có StandardScaler trong ColumnTransformer. Nếu scale ở đây sẽ bị DOUBLE SCALING
    # → Model sẽ scale lại data đã được scale → kết quả prediction SAI!
    # → Nên để scale_method="none" (mặc định) và để model tự scale khi training.
    scaling_stats: dict[str, dict[str, float]] = {}
    if args.scale_method != "none":
        print("=" * 80)
        print("⚠️  CẢNH BÁO NGHIÊM TRỌNG: DOUBLE SCALING DETECTED!")
        print("=" * 80)
        print("⚠️  Bạn đang scale data trong preprocessing!")
        print("⚠️  Model training pipeline (train_level1_rf.py, train_level2_rf.py) đã có StandardScaler.")
        print("⚠️  Model sẽ scale lại data đã được scale → DOUBLE SCALING!")
        print("⚠️  Kết quả: Prediction sẽ SAI hoàn toàn!")
        print("=" * 80)
        print("✓ Khuyến nghị: Sử dụng --scale-method none (mặc định)")
        print("✓ Model sẽ tự scale data khi training với StandardScaler trong pipeline")
        print("=" * 80)
        # Vẫn tiếp tục scale nếu user yêu cầu, nhưng cảnh báo rõ ràng
        numeric_cols = df.select_dtypes(include=["number"]).columns.difference(skip_cols)
        df, scaling_stats = scale_numeric_features(df, numeric_cols, args.scale_method)
        if scaling_stats:
            print(f"⚠️  Đã scale {len(scaling_stats)} cột theo phương pháp {args.scale_method}.")
            print("⚠️  LƯU Ý: Model sẽ scale lại data này → DOUBLE SCALING!")
            print("⚠️  Kết quả prediction sẽ SAI! Vui lòng retrain với --scale-method none!")
    else:
        print("✓ Không scale data trong preprocessing (đúng - model sẽ tự scale khi training).")

    if args.summary:
        print_summary(df, label_col)
        if label_group_col and label_group_col in df.columns:
            print("Phân bố nhãn nhóm:")
            print(df[label_group_col].value_counts(dropna=False).to_string())
            print("===========================================\n")

    # 13. Lưu kết quả tiền xử lý ra pickle, và CSV nếu được yêu cầu.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(output_path)
    print(f"Đã lưu pickle sạch tại: {output_path}")

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Đã lưu CSV sạch tại: {output_csv}")

    if label_mapping:
        print("Bảng mapping nhãn:")
        for code, label in label_mapping.items():
            print(f"  {code}: {label}")

    metadata: dict[str, object] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "label_column": label_col,
        "label_mapping": label_mapping,
        "label_group_column": label_group_col,
        "label_group_mapping": label_group_mapping,
    }
    if label_group_encoded_mapping:
        metadata["label_group_encoded_mapping"] = label_group_encoded_mapping
    metadata.update({
        "sparse_columns_dropped": sparse_dropped,
        "constant_columns_dropped": constant_dropped,
        "duplicates_removed": int(duplicate_count),
        "clip_stats": clip_stats,
        "scaling_method": args.scale_method,
        "scaling_stats": scaling_stats,
        "one_hot_mapping": category_mapping,
        "balance_method": args.balance_method,  # Ghi nhận phương pháp cân bằng được sử dụng.
        "balance_stats": balance_stats,  # Lưu phân bố trước/sau cho mục đích audit.
    })
    if metadata_output:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        with metadata_output.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Đã lưu metadata tiền xử lý tại: {metadata_output}")

    print("Hoàn tất tiền xử lý.")


if __name__ == "__main__":
    main()


