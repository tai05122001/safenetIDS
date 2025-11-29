"""Split dataset into train/val/test and optionally rebalance the train set.

Usage example:
python scripts/split_dataset.py \
    --source dataset_clean.pkl \
    --output-dir dataset/splits \
    --label-column label_encoded \
    --test-size 0.15 \
    --val-size 0.15 \
    --train-min 10000 \
    --train-max 650000
"""
from __future__ import annotations  # Giúp type hint trả về chính module.

import argparse  # Xử lý tham số dòng lệnh.
import json  # Ghi summary ra file JSON.
from pathlib import Path  # Làm việc với đường dẫn.
from typing import Dict, Tuple  # Type hint cho dict / tuple.
import logging  # Ghi log khi cần.

import numpy as np  # Dùng RNG cho bước oversample.
import pandas as pd  # Đọc/ghi DataFrame.
from sklearn.model_selection import train_test_split  # Hàm cắt train/val/test có stratify.


def parse_args() -> argparse.Namespace:
    """Định nghĩa tập tham số CLI cho script."""
    parser = argparse.ArgumentParser(
        description="Chia dataset thành train/val/test và cân bằng train theo ngưỡng min/max."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset_clean.pkl"),
        help="Đường dẫn dataset đầu vào (pickle hoặc CSV).",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label_encoded",
        help="Tên cột nhãn chi tiết (mặc định: label_encoded).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Thư mục lưu các tập sau khi chia. Nếu không chỉ định sẽ tự động chọn theo level.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Tỷ lệ dữ liệu test (mặc định: 0.15).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Tỷ lệ dữ liệu validation (mặc định: 0.15).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed ngẫu nhiên cho phép chia.",
    )
    parser.add_argument(
        "--train-min",
        type=int,
        default=10000,
        help="Ngưỡng tối thiểu mỗi lớp train sau cân bằng (oversample nếu ít hơn).",
    )
    parser.add_argument(
        "--train-max",
        type=int,
        default=650000,
        help="Ngưỡng tối đa mỗi lớp train (undersample nếu nhiều hơn). Mặc định: 650000 cho Level 1 binary classification.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Nếu muốn dùng một phần dữ liệu để thử nghiệm (0 < frac ≤ 1).",
    )
    parser.add_argument(
        "--level",
        choices=[1, 2, 3],
        type=int,
        default=1,
        help="Chọn level split: 1 (binary), 2 (attack types), hoặc 3 (DoS detail). Mặc định: 1.",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Tên nhóm dùng cho level 2 (ví dụ: dos, rareattack). Bắt buộc khi level=2.",
    )
    parser.add_argument(
        "--group-column",
        type=str,
        default="label_group",
        help="Tên cột nhãn nhóm (dùng cho level 1/2, mặc định: label_group).",
    )
    return parser.parse_args()  # Trả về namespace chứa toàn bộ cấu hình.


def load_dataframe(path: Path) -> pd.DataFrame:
    """Đọc dữ liệu từ pickle hoặc CSV."""
    if not path.exists():  # Kiểm tra file tồn tại.
        raise FileNotFoundError(f"Không tìm thấy dataset tại {path}")
    if path.suffix.lower() in {".pkl", ".pickle"}:  # Ưu tiên pickle.
        return pd.read_pickle(path)
    if path.suffix.lower() == ".csv":  # Hỗ trợ CSV lớn.
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Định dạng nguồn không hỗ trợ: {path.suffix}")  # Báo lỗi nếu định dạng khác.


def stratified_split(
    df: pd.DataFrame,
    label_col: str,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia dữ liệu thành train/val/test với stratify."""
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state,
    )

    adjusted_val_ratio = val_size / (1 - test_size)  # Quy đổi tỷ lệ validation trên tập còn lại.
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val_ratio,
        stratify=train_val[label_col],
        random_state=random_state,
    )
    return train, val, test


def rebalance_train(
    train: pd.DataFrame,
    label_col: str,
    min_target: int,
    max_target: int,
    random_state: int,
    label_balance_config: Dict[str, Dict[str, int]] | None = None,
) -> Tuple[pd.DataFrame, Dict[int | str, int], Dict[int | str, int]]:
    """
    Oversample/undersample tập train theo ngưỡng min/max.
    
    Args:
        train: DataFrame chứa dữ liệu train
        label_col: Tên cột label
        min_target: Số mẫu tối thiểu mặc định cho mỗi label
        max_target: Số mẫu tối đa mặc định cho mỗi label
        random_state: Seed để tái lập
        label_balance_config: Dict chứa config riêng cho từng label
            Format: {'Label Name': {'min': 20000, 'max': 30000}, ...}
            Nếu không có config cho label, sẽ dùng min_target/max_target mặc định
    
    Returns:
        Tuple[balanced_df, counts_before, counts_after]
    """
    rng = np.random.default_rng(random_state)  # Generator giúp tái lập.
    before = train[label_col].value_counts().to_dict()  # Lưu phân bố ban đầu.

    balanced_frames = []
    if label_balance_config:
        logging.info(f"[rebalance] label_balance_config keys: {list(label_balance_config.keys())}")
    else:
        logging.info(f"[rebalance] No label_balance_config provided")
    for label_value, subset in train.groupby(label_col):  # Xử lý riêng từng nhãn.
        count = len(subset)
        
        # Lấy config riêng cho label này nếu có
        if label_balance_config and label_value in label_balance_config:
            config = label_balance_config[label_value]
            # Chỉ áp dụng min/max nếu có trong config
            label_min = config.get('min', None)  # None = không oversample
            label_max = config.get('max', None)  # None = không undersample
            logging.info(f"[rebalance] Label {label_value}: count={count}, config found: min={label_min}, max={label_max}")
        else:
            # Nếu không có config riêng → giữ nguyên (không áp dụng min/max)
            label_min = None  # Không oversample
            label_max = None  # Không undersample
            logging.info(f"[rebalance] Label {label_value}: count={count}, no config (keeping original)")
        
        # Chỉ oversample nếu có label_min và count < label_min
        if label_min is not None and count < label_min:
            need = label_min - count
            extra_index = rng.choice(subset.index.to_numpy(), size=need, replace=True)  # Lấy mẫu có lặp.
            subset = pd.concat([subset, train.loc[extra_index]])
            logging.info(f"[rebalance] Label {label_value}: oversampled from {count} to {label_min}")
        # Chỉ undersample nếu có label_max và count > label_max
        elif label_max is not None and count > label_max:
            subset = subset.sample(n=label_max, random_state=random_state, replace=False)
            logging.info(f"[rebalance] Label {label_value}: undersampled from {count} to {label_max}")
        # Nếu không có config (label_min=None, label_max=None) → giữ nguyên (không làm gì)
        balanced_frames.append(subset)

    balanced = pd.concat(balanced_frames, ignore_index=True)  # Ghép các phần đã xử lý.
    balanced = balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)  # Shuffle lại train.
    after = balanced[label_col].value_counts().to_dict()  # Ghi nhận phân bố sau cân bằng.
    return balanced, before, after


def ensure_label_group_column(
    df: pd.DataFrame, label_col: str, group_col: str
) -> str:
    """Đảm bảo tồn tại cột label_group theo mapping chuẩn; trả về tên cột."""
    try:
        return resolve_column(df, group_col)
    except KeyError:
        normalized = group_col.lower()
        lookup = {col.lower(): col for col in df.columns}
        if normalized in lookup:
            return lookup[normalized]

    # Tạo cột mới dựa trên label chi tiết.
    # Chỉ giữ lại: benign, dos, ddos, portscan
    # Đã bỏ: Bot, Infiltration, Heartbleed
    normalized_labels = df[label_col].astype(str).str.strip().str.lower()
    mapping_rules = {
        "benign": "benign",
        "dos hulk": "dos",
        "dos goldeneye": "dos",
        "dos slowloris": "dos",
        "dos slowhttptest": "dos",
        "ddos": "ddos",
        "portscan": "portscan",
        # Bot, Infiltration, Heartbleed đã được loại bỏ
    }
    df[group_col] = normalized_labels.map(mapping_rules).fillna("other")
    return group_col


def ensure_binary_label_column(
    df: pd.DataFrame, group_col: str, binary_col: str = "label_binary_encoded"
) -> str:
    """Tạo cột binary label: 0=benign, 1=attack (gộp dos, ddos, portscan)"""
    if binary_col in df.columns:
        return binary_col
    
    def map_to_binary(group_value):
        if pd.isna(group_value):
            return 0
        group_str = str(group_value).lower().strip()
        if group_str == "benign":
            return 0
        else:  # dos, ddos, portscan, other -> attack
            return 1
    
    df[binary_col] = df[group_col].apply(map_to_binary)
    return binary_col


def ensure_attack_type_label_column(
    df: pd.DataFrame, group_col: str, attack_type_col: str = "label_attack_type_encoded"
) -> str:
    """Tạo cột attack type label: 0=dos, 1=ddos, 2=portscan (chỉ cho attack, benign = -1)"""
    if attack_type_col in df.columns:
        return attack_type_col
    
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
    return attack_type_col


def resolve_column(df: pd.DataFrame, column_name: str) -> str:
    """Tìm tên cột thực tế trong DataFrame (không phân biệt hoa thường)."""
    if column_name in df.columns:  # Nếu tên truyền vào khớp 100% thì trả luôn.
        return column_name
    normalized = column_name.lower()
    lookup = {col.lower(): col for col in df.columns}  # Map để tìm theo lowercase.
    if normalized in lookup:
        return lookup[normalized]
    raise KeyError(f"Không tìm thấy cột '{column_name}' trong dataset.")


def get_dos_balance_config() -> Dict[str, Dict[str, int]]:
    """
    Tạo config cân bằng cho các DoS attack types.
    DoS Hulk chiếm quá nhiều (91.4%), cần giảm xuống 10K để model không bias.
    Các label khác (DoS GoldenEye, DoS slowloris, DoS Slowhttptest) giữ nguyên số lượng.
    
    Returns:
        Dict với format: {'DoS Hulk': {'max': 10000}, ...}
        Chỉ có DoS Hulk trong config, các label khác không có → giữ nguyên
    """
    return {
        'DoS Hulk': {'max': 10000},  # Undersample từ 231K xuống 10K
        # Các biến thể tên (case-insensitive)
        'dos hulk': {'max': 10000},
        # Các label khác (DoS GoldenEye, DoS slowloris, DoS Slowhttptest) 
        # KHÔNG có trong config → sẽ giữ nguyên số lượng (không oversample)
    }


def split_and_save(
    df: pd.DataFrame,
    label_col: str,
    args: argparse.Namespace,
    output_dir: Path,
    context: str,
    label_balance_config: Dict[str, Dict[str, int]] | None = None,
) -> dict:
    """
    Chia dữ liệu theo nhãn, cân bằng train và lưu kết quả.
    
    Args:
        df: DataFrame chứa dữ liệu
        label_col: Tên cột label
        args: Namespace chứa các tham số
        output_dir: Thư mục lưu kết quả
        context: Context string để log
        label_balance_config: Config cân bằng riêng cho từng label (optional)
    """
    output_dir.mkdir(parents=True, exist_ok=True)  # Đảm bảo thư mục đầu ra tồn tại.

    label_counts = df[label_col].value_counts()  # Đếm số mẫu mỗi nhãn.
    rare_labels = label_counts[label_counts < 2].index.tolist()  # Nhãn có < 2 mẫu gây lỗi stratify.
    if rare_labels:
        logging.info(f"[{context}] Rare labels (count < 2) moved directly into train set: {list(rare_labels)}")
    rare_df = df[df[label_col].isin(rare_labels)]  # Tách riêng nhãn quá hiếm.
    df_split = df[~df[label_col].isin(rare_labels)]  # Phần còn lại đủ lớn để stratify.

    if df_split.empty or df_split[label_col].nunique() < 2:
        # Nếu sau khi loại nhãn hiếm mà không còn đủ lớp thì đưa hết vào train.
        logging.warning(f"[{context}] Không thể stratify (quá ít nhãn). Sử dụng toàn bộ dữ liệu cho train.")
        train = df.copy()
        val = df.iloc[0:0].copy()
        test = df.iloc[0:0].copy()
    else:
        train, val, test = stratified_split(
            df_split,
            label_col,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
        )
        if not rare_df.empty:
            train = pd.concat([train, rare_df], ignore_index=True)

    logging.info(
        f"[{context}] Split summary: train={train.shape[0]}, val={val.shape[0]}, test={test.shape[0]}"
    )

    train_balanced, counts_before, counts_after = rebalance_train(
        train,
        label_col,
        min_target=args.train_min,
        max_target=args.train_max,
        random_state=args.random_state,
        label_balance_config=label_balance_config,
    )
    logging.info(f"[{context}] Train distribution before balancing:")
    logging.info(str(counts_before))
    logging.info(f"[{context}] Train distribution after balancing:")
    logging.info(str(counts_after))

    # Lưu từng tập ra file pickle để dùng cho huấn luyện sau.
    train.reset_index(drop=True).to_pickle(output_dir / "train_raw.pkl")
    val.reset_index(drop=True).to_pickle(output_dir / "val.pkl")
    test.reset_index(drop=True).to_pickle(output_dir / "test.pkl")
    train_balanced.reset_index(drop=True).to_pickle(output_dir / "train_balanced.pkl")

    summary = {
        "label_column": label_col,
        "sizes": {
            "train_raw": int(train.shape[0]),
            "train_balanced": int(train_balanced.shape[0]),
            "val": int(val.shape[0]),
            "test": int(test.shape[0]),
        },
        "counts": {
            "train_raw": counts_before,
            "train_balanced": counts_after,
            "val": val[label_col].value_counts().to_dict(),
            "test": test[label_col].value_counts().to_dict(),
        },
    }
    return summary


def main() -> None:
    """Điểm vào chính của script."""
    # Setup logging để tránh lỗi encoding trên Windows
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)8s | %(message)s",
    )
    args = parse_args()  # Đọc cấu hình từ CLI.

    df = load_dataframe(args.source)  # Đọc dataset đầu vào.
    try:
        label_col = resolve_column(df, args.label_column)  # Xác định cột nhãn chi tiết.
    except KeyError:
        # Thử loại bỏ hậu tố _encoded nếu có.
        fallback_candidates = []
        if args.label_column.endswith("_encoded"):
            fallback_candidates.append(args.label_column[:-8])
        if args.label_column.endswith("_group_encoded"):
            fallback_candidates.append(args.label_column.replace("_group_encoded", "_encoded"))
            fallback_candidates.append(args.label_column.replace("_group_encoded", ""))
        for candidate in fallback_candidates:
            try:
                label_col = resolve_column(df, candidate)
                logging.info("Sử dụng cột nhãn fallback: %s", label_col)
                break
            except KeyError:
                continue
        else:
            raise

    if args.sample_frac is not None:  # Cho phép sample bớt dữ liệu khi muốn thử nhanh.
        if not 0 < args.sample_frac <= 1:
            raise ValueError("--sample-frac phải nằm trong (0, 1].")
        df = df.sample(frac=args.sample_frac, random_state=args.random_state).reset_index(drop=True)

    logging.info(f"Dataset: {args.source} | rows={df.shape[0]} | cols={df.shape[1]} | label={label_col}")

    # Chuẩn bị thư mục đầu ra theo level.
    if args.output_dir is None:
        if args.level == 1:
            output_dir = Path("dataset/splits/level1")
        elif args.level == 2:
            output_dir = Path("dataset/splits/level2")
        else:  # level == 3
            if not args.group:
                raise ValueError("Phải cung cấp --group khi level=3.")
            output_dir = Path("dataset/splits/level3") / args.group.lower()
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.level == 1:
        # Level 1: Binary classification (benign vs attack)
        group_col = ensure_label_group_column(df, label_col, args.group_column)
        logging.info(f"Sử dụng cột nhóm: {group_col}")
        
        # Tạo cột binary label cho Level 1
        binary_col = ensure_binary_label_column(df, group_col, "label_binary_encoded")
        logging.info(f"Tạo cột binary label: {binary_col}")
        logging.info(f"Binary distribution: {df[binary_col].value_counts().to_dict()}")
        
        # Tạo cột attack type label cho Level 2 (lưu sẵn để dùng sau)
        attack_type_col = ensure_attack_type_label_column(df, group_col, "label_attack_type_encoded")
        logging.info(f"Tạo cột attack type label: {attack_type_col}")
        logging.info(f"Attack type distribution: {df[attack_type_col].value_counts().to_dict()}")
        
        # Level 1: Chỉ giới hạn benign xuống train_max
        # Cần map từ group_col (benign) sang binary_col (0)
        level1_balance_config = {}
        unique_groups = df[group_col].unique()
        for label_val in unique_groups:
            label_str = str(label_val).lower().strip()
            if label_str == "benign" or label_val == 0:
                # Map sang giá trị binary: benign -> 0
                binary_value = 0
                level1_balance_config[binary_value] = {'max': args.train_max}
                logging.info(f"[level1] Áp dụng train_max={args.train_max} cho benign (group_col={label_val} -> binary_col={binary_value})")
                break
        
        summary = {
            "mode": "level1",
            "source": str(args.source),
            "group_column": group_col,
            "binary_column": binary_col,
            "attack_type_column": attack_type_col,
            "label_column": label_col,
            "train_min": args.train_min,
            "train_max": args.train_max,
            "test_size": args.test_size,
            "val_size": args.val_size,
            "random_state": args.random_state,
            "split": split_and_save(df, binary_col, args, output_dir, context="level1", label_balance_config=level1_balance_config),
        }
    elif args.level == 2:
        # Level 2: Attack types classification (dos, ddos, portscan)
        group_col = ensure_label_group_column(df, label_col, args.group_column)
        logging.info(f"Sử dụng cột nhóm: {group_col}")
        
        # Chỉ lấy các samples là attack (không có benign)
        subset = df[df[group_col].astype(str).str.lower() != "benign"].copy()
        if subset.empty:
            raise ValueError(f"Không tìm thấy dữ liệu attack cho level 2.")
        
        # Tạo attack type label
        attack_type_col = ensure_attack_type_label_column(subset, group_col, "label_attack_type_encoded")
        
        # Chỉ giữ các samples có attack_type >= 0 (loại bỏ -1)
        subset = subset[subset[attack_type_col] >= 0].copy()
        
        if subset.empty:
            raise ValueError(f"Không tìm thấy dữ liệu attack types hợp lệ cho level 2.")
        
        logging.info(f"[level2] Attack types distribution: {subset[attack_type_col].value_counts().to_dict()}")
        
        summary = {
            "mode": "level2",
            "source": str(args.source),
            "group": "attack_types",
            "group_column": group_col,
            "label_column": attack_type_col,
            "train_min": args.train_min,
            "train_max": args.train_max,
            "test_size": args.test_size,
            "val_size": args.val_size,
            "random_state": args.random_state,
            "split": split_and_save(
                subset,
                attack_type_col,
                args,
                output_dir,
                context="level2:attack_types",
                label_balance_config=None,
            ),
        }
    else:  # level == 3
        # Level 3: DoS detail classification (DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest)
        if not args.group:
            raise ValueError("Phải cung cấp --group khi level=3.")
        group_col = ensure_label_group_column(df, label_col, args.group_column)
        target_group = args.group.lower()
        subset = df[df[group_col].astype(str).str.lower() == target_group]
        if subset.empty:
            raise ValueError(f"Không tìm thấy dữ liệu cho nhóm {args.group}.")
        
        # Tạo label_balance_config nếu là nhóm dos
        label_balance_config = None
        if target_group == "dos":
            # Lấy danh sách label thực tế trong dataset
            actual_labels = subset[label_col].unique()
            dos_config = get_dos_balance_config()
            
            # Tìm cột label gốc (không phải encoded) để map số → tên
            # Thử tìm cột 'label' hoặc cột không có '_encoded'
            original_label_col = None
            possible_label_cols = ['label', 'Label', 'Label_original']
            for col in possible_label_cols:
                if col in subset.columns and col != label_col:
                    original_label_col = col
                    break
            
            # Nếu không tìm thấy, thử tìm cột có chứa 'label' nhưng không phải encoded
            if original_label_col is None:
                for col in subset.columns:
                    if 'label' in col.lower() and '_encoded' not in col.lower() and col != label_col:
                        original_label_col = col
                        break
            
            # Map config với tên label thực tế (case-insensitive)
            label_balance_config = {}
            for actual_label in actual_labels:
                # Nếu actual_label là số (encoded), cần map sang tên
                label_name = None
                if isinstance(actual_label, (int, np.integer)) or str(actual_label).isdigit():
                    # Label là số → cần map sang tên
                    if original_label_col and original_label_col in subset.columns:
                        # Lấy một sample với label này để xem tên gốc
                        sample = subset[subset[label_col] == actual_label].iloc[0]
                        label_name = str(sample[original_label_col]).lower().strip()
                    else:
                        # Không tìm thấy cột label gốc, thử dùng label_col trực tiếp
                        # (có thể label_col đã là tên, không phải encoded)
                        label_name = str(actual_label).lower().strip()
                else:
                    # Label đã là tên
                    label_name = str(actual_label).lower().strip()
                
                # Tìm config phù hợp
                for config_key, config_value in dos_config.items():
                    if label_name == config_key.lower():
                        label_balance_config[actual_label] = config_value
                        logging.info(f"[level2:dos] Map label {actual_label} ({label_name}) -> config: {config_value}")
                        break
            
            if label_balance_config:
                logging.info(f"[level2:dos] Sử dụng custom balance config cho DoS labels:")
                for label, config in label_balance_config.items():
                    logging.info(f"  {label}: {config}")
            else:
                logging.warning(f"[level2:dos] ⚠️  Không tìm thấy config cho bất kỳ label nào!")
                logging.info(f"[level2:dos] Actual labels: {actual_labels}")
                logging.info(f"[level2:dos] Dos config keys: {list(dos_config.keys())}")
                if original_label_col:
                    logging.info(f"[level2:dos] Original label column: {original_label_col}")
                    # Log một vài samples để debug
                    for label_val in actual_labels[:3]:
                        sample = subset[subset[label_col] == label_val].iloc[0]
                        logging.info(f"[level2:dos]   Label {label_val} -> original: {sample[original_label_col]}")
        
        summary = {
            "mode": "level3",
            "source": str(args.source),
            "group": args.group,
            "group_column": group_col,
            "label_column": label_col,
            "train_min": args.train_min,
            "train_max": args.train_max,
            "test_size": args.test_size,
            "val_size": args.val_size,
            "random_state": args.random_state,
            "split": split_and_save(
                subset,
                label_col,
                args,
                output_dir,
                context=f"level3:{args.group}",
                label_balance_config=label_balance_config,
            ),
        }

    # Ghi lại toàn bộ cấu hình & thống kê để tiện theo dõi.
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info(f"Tổng hợp cấu hình đã lưu tại {output_dir.resolve() / 'split_summary.json'}")


if __name__ == "__main__":
    main()
