"""Split dataset into train/val/test and optionally rebalance the train set.

Usage example:
python scripts/split_dataset.py \
    --source dataset_clean.pkl \
    --output-dir dataset/splits \
    --label-column label_encoded \
    --test-size 0.15 \
    --val-size 0.15 \
    --train-min 10000 \
    --train-max 200000
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
        default=200000,
        help="Ngưỡng tối đa mỗi lớp train (undersample nếu nhiều hơn).",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Nếu muốn dùng một phần dữ liệu để thử nghiệm (0 < frac ≤ 1).",
    )
    parser.add_argument(
        "--level",
        choices=[1, 2],
        type=int,
        default=1,
        help="Chọn level split: 1 (label_group) hoặc 2 (nhóm cụ thể). Mặc định: 1.",
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
) -> Tuple[pd.DataFrame, Dict[int | str, int], Dict[int | str, int]]:
    """Oversample/undersample tập train theo ngưỡng min/max."""
    rng = np.random.default_rng(random_state)  # Generator giúp tái lập.
    before = train[label_col].value_counts().to_dict()  # Lưu phân bố ban đầu.

    balanced_frames = []
    for label_value, subset in train.groupby(label_col):  # Xử lý riêng từng nhãn.
        count = len(subset)
        if count < min_target:  # Thiếu mẫu -> oversample.
            need = min_target - count
            extra_index = rng.choice(subset.index.to_numpy(), size=need, replace=True)  # Lấy mẫu có lặp.
            subset = pd.concat([subset, train.loc[extra_index]])
        elif count > max_target:  # Dư mẫu -> undersample.
            subset = subset.sample(n=max_target, random_state=random_state, replace=False)
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
    normalized_labels = df[label_col].astype(str).str.strip().str.lower()
    mapping_rules = {
        "benign": "benign",
        "dos hulk": "dos",
        "dos goldeneye": "dos",
        "dos slowloris": "dos",
        "dos slowhttptest": "dos",
        "ddos": "ddos",
        "portscan": "portscan",
        "bot": "rare_attack",
        "infiltration": "rare_attack",
        "heartbleed": "rare_attack",
    }
    df[group_col] = normalized_labels.map(mapping_rules).fillna("other")
    return group_col


def resolve_column(df: pd.DataFrame, column_name: str) -> str:
    """Tìm tên cột thực tế trong DataFrame (không phân biệt hoa thường)."""
    if column_name in df.columns:  # Nếu tên truyền vào khớp 100% thì trả luôn.
        return column_name
    normalized = column_name.lower()
    lookup = {col.lower(): col for col in df.columns}  # Map để tìm theo lowercase.
    if normalized in lookup:
        return lookup[normalized]
    raise KeyError(f"Không tìm thấy cột '{column_name}' trong dataset.")


def split_and_save(
    df: pd.DataFrame,
    label_col: str,
    args: argparse.Namespace,
    output_dir: Path,
    context: str,
) -> dict:
    """Chia dữ liệu theo nhãn, cân bằng train và lưu kết quả."""
    output_dir.mkdir(parents=True, exist_ok=True)  # Đảm bảo thư mục đầu ra tồn tại.

    label_counts = df[label_col].value_counts()  # Đếm số mẫu mỗi nhãn.
    rare_labels = label_counts[label_counts < 2].index.tolist()  # Nhãn có < 2 mẫu gây lỗi stratify.
    if rare_labels:
        print(f"[{context}] Rare labels (count < 2) moved directly into train set: {list(rare_labels)}")
    rare_df = df[df[label_col].isin(rare_labels)]  # Tách riêng nhãn quá hiếm.
    df_split = df[~df[label_col].isin(rare_labels)]  # Phần còn lại đủ lớn để stratify.

    if df_split.empty or df_split[label_col].nunique() < 2:
        # Nếu sau khi loại nhãn hiếm mà không còn đủ lớp thì đưa hết vào train.
        print(f"[{context}] Không thể stratify (quá ít nhãn). Sử dụng toàn bộ dữ liệu cho train.")
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

    print(
        f"[{context}] Split summary: train={train.shape[0]}, val={val.shape[0]}, test={test.shape[0]}"
    )

    train_balanced, counts_before, counts_after = rebalance_train(
        train,
        label_col,
        min_target=args.train_min,
        max_target=args.train_max,
        random_state=args.random_state,
    )
    print(f"[{context}] Train distribution before balancing:")
    print(counts_before)
    print(f"[{context}] Train distribution after balancing:")
    print(counts_after)

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

    print(f"Dataset: {args.source} | rows={df.shape[0]} | cols={df.shape[1]} | label={label_col}")

    # Chuẩn bị thư mục đầu ra theo level.
    if args.output_dir is None:
        if args.level == 1:
            output_dir = Path("dataset/splits/level1")
        else:
            if not args.group:
                raise ValueError("Phải cung cấp --group khi level=2.")
            output_dir = Path("dataset/splits/level2") / args.group.lower()
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.level == 1:
        group_col = ensure_label_group_column(df, label_col, args.group_column)
        print(f"Sử dụng cột nhóm: {group_col}")
        summary = {
            "mode": "level1",
            "source": str(args.source),
            "group_column": group_col,
            "label_column": label_col,
            "train_min": args.train_min,
            "train_max": args.train_max,
            "test_size": args.test_size,
            "val_size": args.val_size,
            "random_state": args.random_state,
            "split": split_and_save(df, group_col, args, output_dir, context="level1"),
        }
    else:
        if not args.group:
            raise ValueError("Phải cung cấp --group khi level=2.")
        group_col = ensure_label_group_column(df, label_col, args.group_column)
        target_group = args.group.lower()
        subset = df[df[group_col].astype(str).str.lower() == target_group]
        if subset.empty:
            raise ValueError(f"Không tìm thấy dữ liệu cho nhóm {args.group}.")
        summary = {
            "mode": "level2",
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
                context=f"level2:{args.group}",
            ),
        }

    # Ghi lại toàn bộ cấu hình & thống kê để tiện theo dõi.
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Tổng hợp cấu hình đã lưu tại {output_dir.resolve() / 'split_summary.json'}")


if __name__ == "__main__":
    main()
