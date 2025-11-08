"""
Utility: đọc file dataset CSV, cache sang pickle và in thông tin nhanh.

Ví dụ chạy:
python scripts/load_dataset.py --path dataset.csv --head 5 --pickle dataset.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Định nghĩa tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Đọc nhanh dataset CSV, hỗ trợ cache pickle.")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("dataset.csv"),
        help="Đường dẫn đến file CSV (mặc định: dataset.csv).",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Số dòng đầu tiên cần hiển thị (mặc định 5).",
    )
    parser.add_argument(
        "--pickle",
        type=Path,
        default=Path("dataset.pkl"),
        help="Đường dẫn file pickle để cache dữ liệu (mặc định: dataset.pkl).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Bỏ qua cache pickle và đọc lại từ CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.path.exists() and (not args.pickle.exists() or args.refresh):
        raise FileNotFoundError(f"Không tìm thấy file: {args.path}")

    df: pd.DataFrame | None = None
    source_path: Path

    if args.pickle.exists() and not args.refresh:
        print(f"Loading dataset từ cache pickle: {args.pickle}")
        df = pd.read_pickle(args.pickle)
        source_path = args.pickle
    else:
        print(f"Loading dataset từ CSV: {args.path}")
        df = pd.read_csv(args.path, low_memory=False)
        try:
            df.to_pickle(args.pickle)
            print(f"Đã lưu cache pickle tại: {args.pickle}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Không thể lưu pickle: {exc}")
        source_path = args.path

    # In thông tin tổng quan.
    print(f"Loaded dataset từ: {source_path}")
    print(f"Shape          : {df.shape[0]} rows x {df.shape[1]} columns")
    print("Columns        :", ", ".join(df.columns.tolist()))

    # In thống kê cơ bản (các cột số).
    print("\nMô tả thống kê (numeric columns):")
    print(df.describe(include="number"))

    # Hiển thị các dòng đầu tiên để kiểm tra dữ liệu.
    print(f"\nTop {args.head} rows:")
    print(df.head(args.head))


if __name__ == "__main__":
    main()


