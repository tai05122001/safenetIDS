#!/usr/bin/env python3
"""
Safenet IDS - Data Preprocessing Service
Dịch vụ tiền xử lý dữ liệu network từ Kafka topic 'raw_data_event'

Chức năng chính:
- Đọc dữ liệu thô từ Kafka raw_data_event (consumer)
- Áp dụng các bước preprocessing giống HOÀN TOÀN với scripts/preprocess_dataset.py
- Chuẩn hóa tên cột, xử lý missing values, encoding, scaling
- Gửi dữ liệu đã xử lý đến topic preprocess_data

Luồng xử lý:
1. Nhận message từ raw_data_event (consumer)
2. Parse JSON thành Python dict
3. Áp dụng preprocessing pipeline tùy theo model_type:
   - Normalize column names
   - Replace inf/-inf với NaN
   - Convert numeric columns
   - Fill missing values (median/mode)
   - Encode label và tạo label_group
   - Drop constant columns (skip cho single record)
   - Clip outliers IQR
   - Scale tùy model_type:
     * Random Forest: Scale ở preprocessing (standard scaling)
     * CNN+LSTM: KHÔNG scale (model sẽ tự scale)
4. Gửi kết quả đến preprocess_data
5. Log và monitoring

Lưu ý: Scaling được bỏ để tránh double scaling vì model Level 1 đã có StandardScaler
"""

# Import các thư viện cần thiết
import json  # Parse JSON messages từ Kafka
import logging  # Logging hoạt động service
import re  # Regular expressions cho column name normalization
import numpy as np  # Thư viện tính toán số học
import pandas as pd  # Data manipulation và analysis
from pathlib import Path  # File path handling
from typing import Iterable, Dict, Any, Tuple  # Type hints
from kafka import KafkaConsumer, KafkaProducer  # Kafka clients
from datetime import datetime, timedelta  # Timestamp handling

# Cấu hình logging system
# - Level INFO: Ghi các event quan trọng
# - Format: timestamp, logger name, level, message
# - Handlers: File log và console output
# - Encoding UTF-8 để hỗ trợ ký tự tiếng Việt trên Windows
import os
os.makedirs('services/logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('services/logs/data_preprocessing.log', encoding='utf-8'),  # UTF-8 encoding cho tiếng Việt
        logging.StreamHandler()  # Console output for debugging
    ]
)

# Tạo logger instance cho service này
logger = logging.getLogger('DataPreprocessing')

class DataPreprocessingService:
    """
    Lớp chính của Data Preprocessing Service

    Trách nhiệm:
    - Quản lý kết nối Kafka consumer và producer
    - Triển khai preprocessing pipeline cho dữ liệu streaming
    - Xử lý lỗi và đảm bảo data quality
    - Monitoring và logging performance

    Pipeline preprocessing (giống hệt scripts/preprocess_dataset.py, TRỪ scaling):
    1. Normalize column names
    2. Replace inf/-inf với NaN
    3. Convert numeric columns
    4. Fill missing values (median cho số, mode cho categorical)
    5. Remove rows with missing labels
    6. Encode label (tạo label_encoded)
    7. Add label_group column và encode (tạo label_group_encoded)
    8. Add binary label column (tạo label_binary_encoded: 0=benign, 1=attack)
    9. Add attack type label column (tạo label_attack_type_encoded: 0=dos, 1=ddos, 2=portscan, -1=benign)
    10. Drop constant columns (skip cho single record)
    11. IQR outlier clipping
    12. KHÔNG scale (để model tự scale qua StandardScaler trong pipeline)
    
    Lý do bỏ scaling tùy theo model_type:
    - Random Forest: Scale ở preprocessing vì model không có internal scaler
    - CNN+LSTM: Không scale ở preprocessing vì model có StandardScaler trong pipeline
    - Nếu scale ở đây sẽ bị double scaling → kết quả sai
    """

    def __init__(self,
                 kafka_bootstrap_servers='localhost:9092',
                 input_topic='raw_data_event',
                 output_topic='preprocess_data',
                 model_type='random_forest'):
        """
        Khởi tạo Data Preprocessing Service

        Args:
            kafka_bootstrap_servers: Kafka bootstrap servers
            input_topic: Topic để đọc dữ liệu thô
            output_topic: Topic để gửi dữ liệu đã xử lý
            model_type: Loại model ('random_forest' hoặc 'cnn_lstm')
        """
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.consumer = None
        self.producer = None
        self.kafka_servers = kafka_bootstrap_servers
        self.model_type = model_type
        self.is_running = False

        # Metadata để tái sử dụng logic preprocessing
        self.preprocessing_metadata = {}
        
        # Summary statistics
        self.processed_count = 0
        self.label_summary = {}
        self.label_group_summary = {}
        self.error_count = 0
        
        # Timestamp khi service khởi động để chỉ xử lý message mới
        self.service_start_time = None
        # Lưu partition assignment để detect rebalance
        self.last_partition_assignment = set()
        # Lưu timestamp của message đầu tiên được xử lý
        # Điều này đảm bảo chỉ xử lý message có timestamp >= message đầu tiên
        self.first_message_timestamp = None

        # Khởi tạo consumer và producer
        self._init_consumer()
        self._init_producer()

    def _init_consumer(self):
        """Khởi tạo Kafka consumer"""
        try:
            self.consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.kafka_servers,
                group_id='safenet-ids-preprocessing-group',
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                key_deserializer=lambda x: x.decode('utf-8') if x else None,
                auto_offset_reset='latest',  # Bắt đầu từ message mới nhất khi group mới được tạo
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                session_timeout_ms=30000,
                heartbeat_interval_ms=3000,
                max_poll_records=100
            )
            logger.info(f"Kafka consumer initialized for topic: {self.input_topic}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            raise

    def _init_producer(self):
        """Khởi tạo Kafka producer"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: str(k).encode('utf-8'),
                acks='all',
                retries=3,
                linger_ms=5,
                batch_size=32768,
                buffer_memory=67108864
            )
            logger.info(f"Kafka producer initialized for topic: {self.output_topic}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise

    @staticmethod
    def normalize_column(name: str) -> str:
        """
        Chuẩn hóa tên cột về dạng snake_case chữ thường, bỏ ký tự đặc biệt.
        Logic giống hệt scripts/preprocess_dataset.py
        """
        cleaned = name.strip()  # Loại bỏ khoảng trắng ở hai đầu.
        cleaned = re.sub(r"[^\w]+", "_", cleaned, flags=re.UNICODE)  # Thay ký tự đặc biệt bằng dấu gạch dưới.
        cleaned = re.sub(r"_+", "_", cleaned)  # Gom các dấu '_' liên tiếp về một dấu.
        return cleaned.strip("_").lower()  # Xóa '_' dư thừa ở biên và đổi về chữ thường.

    @staticmethod
    def convert_numeric(df: pd.DataFrame, skip_cols: Iterable[str]) -> pd.DataFrame:
        """
        Cố gắng ép các cột sang số (trừ những cột bị loại trừ).
        Logic giống hệt scripts/preprocess_dataset.py
        """
        for col in df.columns:
            if col in skip_cols:
                continue  # Không động vào các cột cần giữ nguyên (ví dụ cột nhãn).
            if pd.api.types.is_numeric_dtype(df[col]):
                continue  # Nếu đã là số thì không cần xử lý thêm.
            coerced = pd.to_numeric(df[col], errors="coerce")  # Thử chuyển về kiểu số; lỗi sẽ thành NaN.
            df[col] = coerced  # Gán lại cột đã được xử lý.
        return df

    @staticmethod
    def fill_missing_values(df: pd.DataFrame, skip_cols: Iterable[str]) -> pd.DataFrame:
        """
        Điền giá trị thiếu: median cho cột số, mode cho cột phân loại.
        Logic giống hệt scripts/preprocess_dataset.py
        """
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

    @staticmethod
    def encode_label(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, Dict[int, str]]:
        """
        Mã hóa cột nhãn sang số và trả về mapping ngược.
        Logic giống hệt scripts/preprocess_dataset.py
        """
        if label_col not in df.columns:
            logger.info(f"Không tìm thấy cột nhãn {label_col}, bỏ qua bước mã hóa.")
            return df, {}

        label_series = df[label_col].astype(str).str.strip()  # Đồng nhất kiểu dữ liệu và bỏ khoảng trắng.
        df[label_col] = label_series  # Cập nhật lại cột gốc sau khi làm sạch.
        codes, uniques = pd.factorize(label_series, sort=True)  # Mã hóa nhãn thành số nguyên.
        df[f"{label_col}_encoded"] = codes  # Thêm cột nhãn đã mã hóa.
        # Ensure mapping uses pure Python types
        uniques_list = [str(label) for label in uniques]  # Convert to list of strings
        mapping = {int(code): uniques_list[code] for code in range(len(uniques_list))}  # Bảng tra cứu để giải mã lại.
        return df, mapping

    @staticmethod
    def add_label_group_column(
        df: pd.DataFrame,
        source_col: str,
        group_col: str,
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Tạo cột gom nhóm nhãn theo logic tùy chỉnh.
        Logic giống hệt scripts/preprocess_dataset.py
        """
        if source_col not in df.columns:
            logger.info(f"Không tìm thấy cột nhãn {source_col} để tạo nhóm.")
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
            # Bot, Infiltration, Heartbleed đã được loại bỏ
        }

        default_group = "other"  # Phân loại mặc định cho nhãn chưa định nghĩa.
        mapping_report: Dict[str, str] = {}

        cleaned_series = df[source_col].astype(str).str.strip()
        # Xây dựng mapping thực tế từ nhãn gốc -> nhóm.
        for original_label in cleaned_series.unique():
            normalized = str(original_label).lower()
            group_name = group_rules.get(normalized, default_group)
            mapping_report[str(original_label)] = group_name

        group_series = cleaned_series.map(mapping_report)
        df[group_col] = group_series  # Gán vào DataFrame.
        return df, mapping_report

    @staticmethod
    def add_binary_label_column(
        df: pd.DataFrame,
        group_col: str,
        binary_col: str = "label_binary_encoded"
    ) -> pd.DataFrame:
        """
        Tạo cột binary label: 0=benign, 1=attack (gộp dos, ddos, portscan)
        Logic giống hệt scripts/preprocess_dataset.py
        """
        if group_col not in df.columns:
            logger.info(f"Không tìm thấy cột nhóm {group_col} để tạo binary label.")
            return df
        
        if binary_col in df.columns:
            logger.debug(f"Cột {binary_col} đã tồn tại, bỏ qua.")
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
        logger.debug(f"Đã tạo cột binary label: {binary_col}")
        logger.debug(f"Binary distribution: {df[binary_col].value_counts().to_dict()}")
        return df

    @staticmethod
    def add_attack_type_label_column(
        df: pd.DataFrame,
        group_col: str,
        attack_type_col: str = "label_attack_type_encoded"
    ) -> pd.DataFrame:
        """
        Tạo cột attack type label: 0=dos, 1=ddos, 2=portscan (chỉ cho attack, benign = -1)
        Logic giống hệt scripts/preprocess_dataset.py
        """
        if group_col not in df.columns:
            logger.info(f"Không tìm thấy cột nhóm {group_col} để tạo attack type label.")
            return df
        
        if attack_type_col in df.columns:
            logger.debug(f"Cột {attack_type_col} đã tồn tại, bỏ qua.")
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
        logger.debug(f"Đã tạo cột attack type label: {attack_type_col}")
        logger.debug(f"Attack type distribution: {df[attack_type_col].value_counts().to_dict()}")
        return df

    @staticmethod
    def drop_sparse_columns(df: pd.DataFrame, min_non_null_ratio: float, skip_cols: Iterable[str]) -> pd.DataFrame:
        """
        Loại bỏ các cột có quá nhiều giá trị null
        """
        df_copy = df.copy()
        cols_to_drop = []

        for col in df_copy.columns:
            if col not in skip_cols:
                non_null_ratio = df_copy[col].notnull().mean()
                if non_null_ratio < min_non_null_ratio:
                    cols_to_drop.append(col)
                    logger.info(f"Dropping sparse column {col} (non-null ratio: {non_null_ratio:.2f})")

        df_copy = df_copy.drop(columns=cols_to_drop)
        return df_copy

    @staticmethod
    def drop_constant_columns(
        df: pd.DataFrame, skip_cols: Iterable[str]
    ) -> Tuple[pd.DataFrame, list]:
        """
        Loại bỏ các cột chỉ có một giá trị (bao gồm cả NaN).
        Logic giống hệt scripts/preprocess_dataset.py
        """
        dropped: list = []
        for col in df.columns:
            if col in skip_cols:
                continue
            if df[col].nunique(dropna=False) <= 1:  # Cột có 0-1 giá trị => không hữu dụng.
                dropped.append(col)
        if dropped:
            df = df.drop(columns=dropped)
        return df, dropped

    @staticmethod
    def clip_outliers_iqr(
        df: pd.DataFrame, numeric_cols: Iterable[str], factor: float
    ) -> Dict[str, Dict[str, float]]:
        """
        Clip ngoại lệ dựa trên khoảng tứ phân vị (IQR).
        Logic giống hệt scripts/preprocess_dataset.py
        """
        if factor <= 0:
            return {}
        clip_info: Dict[str, Dict[str, float]] = {}
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

    @staticmethod
    def scale_numeric_features(
        df: pd.DataFrame, numeric_cols: Iterable[str], method: str
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """
        Chuẩn hóa dữ liệu số theo phương pháp lựa chọn.
        Logic giống hệt scripts/preprocess_dataset.py
        """
        stats: Dict[str, Dict[str, float]] = {}
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

    def preprocess_single_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tiền xử lý một record đơn lẻ theo logic giống hệt scripts/preprocess_dataset.py

        Args:
            record: Dictionary chứa dữ liệu network thô

        Returns:
            Dictionary chứa dữ liệu đã tiền xử lý
        """
        try:
            # Chuyển thành DataFrame để xử lý
            df = pd.DataFrame([record])

            # 1. Chuẩn hóa tên cột để dễ thao tác ở các bước sau
            original_columns = list(df.columns)
            df.columns = [self.normalize_column(col) for col in df.columns]
            label_col = self.normalize_column('Label')  # Tên cột nhãn đã chuẩn hóa
            
            # Debug: Log column normalization
            logger.debug(f"Original columns: {original_columns[:10]}...")  # First 10
            logger.debug(f"Normalized columns: {list(df.columns)[:10]}...")  # First 10
            logger.debug(f"Looking for label column: '{label_col}'")
            logger.debug(f"Label column found: {label_col in df.columns}")
            if label_col in df.columns:
                logger.debug(f"Label values in record: {df[label_col].unique()}")

            # 2. Thay thế vô hạn bằng NaN để có thể xử lý thiếu nhất quán
            df = df.replace([np.inf, -np.inf], np.nan)

            # 3. Ép các cột có thể sang kiểu số (trừ cột nhãn)
            df = self.convert_numeric(df, skip_cols={label_col})

            # 4. Điền giá trị thiếu bằng thống kê phù hợp
            df = self.fill_missing_values(df, skip_cols={label_col})

            # 5. Loại bỏ dòng thiếu nhãn (nếu có)
            if label_col in df.columns:
                missing_labels = df[label_col].isna().sum()
                if missing_labels:
                    df = df[df[label_col].notna()].reset_index(drop=True)
                    logger.info(f"Đã loại bỏ {missing_labels} dòng thiếu nhãn ({label_col}).")

            # 6. Mã hóa nhãn (nếu tồn tại)
            label_mapping: Dict[int, str] = {}
            if label_col in df.columns:
                df, label_mapping = self.encode_label(df, label_col)

            # 7. Tạo cột label_group
            label_group_col = 'label_group'
            label_group_mapping: Dict[str, str] = {}
            label_group_encoded_mapping: Dict[int, str] = {}
            if label_col in df.columns:
                df, label_group_mapping = self.add_label_group_column(df, label_col, label_group_col)
                logger.debug(f"Label group mapping: {label_group_mapping}")
                if label_group_col in df.columns:
                    logger.debug(f"Label group values: {df[label_group_col].unique()}")
                df, label_group_encoded_mapping = self.encode_label(df, label_group_col)
                logger.debug(f"Label group encoded mapping: {label_group_encoded_mapping}")
                
                # Tạo binary label và attack type label cho Level 1 và Level 2
                # Không cần encode vì chúng đã là số (0/1 hoặc 0/1/2/-1)
                binary_col = self.normalize_column("label_binary_encoded")
                attack_type_col = self.normalize_column("label_attack_type_encoded")
                df = self.add_binary_label_column(df, label_group_col, binary_col)
                df = self.add_attack_type_label_column(df, label_group_col, attack_type_col)
                logger.debug(f"Đã tạo binary label ({binary_col}) và attack type label ({attack_type_col})")
            else:
                logger.warning(f"Label column '{label_col}' not found, cannot create label_group")

            # Tập các cột cần bỏ qua (giữ nguyên) ở những bước xử lý khác
            skip_cols = {label_col}
            if f"{label_col}_encoded" in df.columns:
                skip_cols.add(f"{label_col}_encoded")
            if label_group_col in df.columns:
                skip_cols.update({label_group_col, f"{label_group_col}_encoded"})
            # Thêm binary và attack type columns vào skip_cols (không có _encoded vì chúng đã là số)
            binary_col = self.normalize_column("label_binary_encoded")
            attack_type_col = self.normalize_column("label_attack_type_encoded")
            if binary_col in df.columns:
                skip_cols.add(binary_col)
            if attack_type_col in df.columns:
                skip_cols.add(attack_type_col)
            skip_cols.update(['timestamp', 'source_ip', 'destination_ip'])

            # 8. Loại bỏ các cột có quá nhiều giá trị thiếu (skip cho single record)
            # df, sparse_dropped = self.drop_sparse_columns(df, 0.5, skip_cols)

            # 9. Loại bỏ các cột không mang thông tin (chỉ có một giá trị)
            # LƯU Ý: Với single record, tất cả cột đều constant → KHÔNG drop để giữ features cho model
            constant_dropped: list = []
            # Chỉ drop constant columns nếu có nhiều records (batch processing)
            # Với streaming single record, skip bước này để giữ tất cả features
            # df, constant_dropped = self.drop_constant_columns(df, skip_cols)
            # if constant_dropped:
            #     logger.info(f"Đã loại bỏ {len(constant_dropped)} cột constant: {constant_dropped}")
            logger.info("Bỏ qua drop constant columns cho single record - giữ tất cả features cho model")

            # 10. Chuẩn hóa ngoại lệ (outlier) bằng cách clip theo khoảng IQR
            clip_stats: Dict[str, Dict[str, float]] = {}
            numeric_cols = df.select_dtypes(include=["number"]).columns.difference(skip_cols)
            clip_stats = self.clip_outliers_iqr(df, numeric_cols, 1.5)
            if clip_stats:
                logger.info(f"Đã clip ngoại lệ cho {len(clip_stats)} cột theo IQR (factor=1.5).")

            # 11. Scale tùy theo model_type
            scaling_stats: Dict[str, Dict[str, float]] = {}

            if self.model_type == "random_forest":
                # Random Forest: Scale ở preprocessing vì model không có internal scaler
                logger.info("Scale data cho Random Forest (model không có internal scaler)")
                df, scaling_stats = self.scale_numeric_features(df, numeric_cols, "standard")
                if scaling_stats:
                    logger.info(f"Đã scale {len(scaling_stats)} cột theo standard scaling cho Random Forest")

            elif self.model_type == "cnn_lstm":
                # CNN+LSTM: Không scale ở preprocessing vì model có StandardScaler trong pipeline
                logger.info("Bỏ qua scaling cho CNN+LSTM - Model sẽ tự scale khi predict")
                scaling_stats = {}  # Empty dict for CNN+LSTM

            else:
                # Default: Không scale để an toàn
                logger.info(f"Bỏ qua scaling cho {self.model_type} - Model sẽ tự scale khi predict")
                scaling_stats = {}  # Empty dict for default

            # Debug: Log số lượng features sau preprocessing
            numeric_features = df.select_dtypes(include=["number"]).columns.difference(skip_cols)
            logger.debug(f"Total numeric features after preprocessing: {len(numeric_features)}")
            logger.debug(f"Total columns in processed record: {len(df.columns)}")
            if label_col in df.columns:
                label_val = df[label_col].iloc[0]
                label_group_val = df.get(label_group_col, 'N/A').iloc[0] if label_group_col in df.columns else 'N/A'
                binary_val = df.get(binary_col, 'N/A').iloc[0] if binary_col in df.columns else 'N/A'
                attack_type_val = df.get(attack_type_col, 'N/A').iloc[0] if attack_type_col in df.columns else 'N/A'
                logger.info(f"Processed record - Label: {label_val}, Label group: {label_group_val}, Binary: {binary_val}, Attack type: {attack_type_val}")
            
            # Chuyển về dictionary và convert tất cả pandas/numpy types thành Python native types
            def convert_to_json_serializable(obj):
                """Convert pandas/numpy objects to JSON serializable Python types"""
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(obj, 'tolist'):  # pandas Index, Series, etc.
                    try:
                        return obj.tolist()
                    except:
                        return str(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(item) for item in obj]
                elif pd.isna(obj):
                    return None
                else:
                    return obj

            processed_record = {k: convert_to_json_serializable(v) for k, v in df.iloc[0].to_dict().items()}

            # Thêm metadata preprocessing
            processed_record['preprocessing_timestamp'] = datetime.now().isoformat()

            # Ensure all metadata components are JSON serializable first
            safe_label_mapping = convert_to_json_serializable(label_mapping)
            safe_label_group_mapping = convert_to_json_serializable(label_group_mapping)
            safe_label_group_encoded_mapping = convert_to_json_serializable(label_group_encoded_mapping)
            safe_clip_stats = convert_to_json_serializable(clip_stats)
            safe_scaling_stats = convert_to_json_serializable(scaling_stats)  # Ensure JSON serializable
            safe_constant_columns_dropped = convert_to_json_serializable(constant_dropped)

            preprocessing_metadata = {
                'model_type': self.model_type,
                'label_mapping': safe_label_mapping,
                'label_group_mapping': safe_label_group_mapping,
                'label_group_encoded_mapping': safe_label_group_encoded_mapping,
                'clip_stats': safe_clip_stats,
                'scaling_stats': safe_scaling_stats,
                'constant_columns_dropped': safe_constant_columns_dropped,
                'processed_columns': [str(col) for col in df.columns],
                'numeric_features_count': len(numeric_features)
            }

            # Thêm thông tin về binary và attack type labels nếu có
            if binary_col in df.columns:
                preprocessing_metadata['binary_label_column'] = binary_col
                preprocessing_metadata['binary_label_value'] = int(df[binary_col].iloc[0]) if pd.notna(df[binary_col].iloc[0]) else None
            if attack_type_col in df.columns:
                preprocessing_metadata['attack_type_label_column'] = attack_type_col
                preprocessing_metadata['attack_type_label_value'] = int(df[attack_type_col].iloc[0]) if pd.notna(df[attack_type_col].iloc[0]) else None

            processed_record['preprocessing_metadata'] = preprocessing_metadata

            return processed_record

        except Exception as e:
            logger.error(f"Error preprocessing record: {e}")
            # Trả về record gốc với flag error
            record['preprocessing_error'] = str(e)
            record['preprocessing_timestamp'] = datetime.now().isoformat()
            # Ensure record is JSON serializable
            def convert_to_json_serializable(obj):
                """Convert pandas/numpy objects to JSON serializable Python types"""
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(item) for item in obj]
                elif pd.isna(obj):
                    return None
                else:
                    return obj

            return {k: convert_to_json_serializable(v) for k, v in record.items()}

    def send_processed_data(self, data: Dict[str, Any], original_key: str = None):
        """
        Gửi dữ liệu đã xử lý đến Kafka topic

        Args:
            data: Dữ liệu đã tiền xử lý
            original_key: Key từ message gốc
        """
        try:
            key = original_key or data.get('timestamp', str(datetime.now().timestamp()))

            # Validate JSON serialization before sending
            try:
                import json
                json.dumps(data)
            except TypeError as json_error:
                logger.error(f"JSON serialization error: {json_error}")
                logger.error("Skipping message due to serialization issue")
                return  # Don't send if serialization fails

            future = self.producer.send(self.output_topic, value=data, key=key)
            record_metadata = future.get(timeout=10)

            logger.info(f"Sent processed data to {record_metadata.topic} "
                       f"partition {record_metadata.partition} "
                       f"offset {record_metadata.offset}")

        except Exception as e:
            logger.error(f"Failed to send processed data: {e}")
            # Log the problematic data structure
            import json
            try:
                json.dumps(data)
            except Exception as json_error:
                logger.error(f"JSON serialization failed: {json_error}")
                # Try to identify the problematic part
                for k, v in data.items():
                    try:
                        json.dumps(v)
                    except:
                        logger.error(f"Problematic field: {k} = {type(v)}")

    def start_processing(self):
        """Bắt đầu quá trình xử lý dữ liệu"""
        self.is_running = True
        
        # Reset các biến thống kê khi service khởi động
        # Điều này đảm bảo không tích lũy số liệu từ các lần chạy trước
        self.processed_count = 0
        self.label_summary = {}
        self.label_group_summary = {}
        self.error_count = 0
        logger.info("Reset statistics counters - starting fresh count")
        
        logger.info(f"Starting data preprocessing service: {self.input_topic} -> {self.output_topic}")

        # Reset offset về cuối topic để chỉ đọc message mới
        # Điều này đảm bảo không đọc message cũ từ lần chạy trước
        try:
            logger.info("Resetting consumer offset to end of topic to skip old messages...")
            # Poll một lần để trigger partition assignment
            self.consumer.poll(timeout_ms=1000)
            
            # Lấy danh sách partitions đã được assign
            partitions = self.consumer.assignment()
            
            if partitions:
                # Lưu timestamp NGAY TRƯỚC KHI seek to end
                # Điều này đảm bảo chỉ xử lý message được gửi sau khi service đã sẵn sàng
                self.service_start_time = datetime.now()
                logger.info(f"Service ready time (before seek to end): {self.service_start_time.isoformat()}")
                
                # Seek về cuối mỗi partition để chỉ đọc message mới
                self.consumer.seek_to_end(*partitions)
                logger.info(f"Seeked to end for {len(partitions)} partitions - will only process new messages")
                # Khởi tạo last_partition_assignment
                self.last_partition_assignment = partitions.copy()
            else:
                logger.warning("No partitions assigned yet, offset reset skipped")
                self.last_partition_assignment = set()
                # Vẫn lưu timestamp để kiểm tra
                self.service_start_time = datetime.now()
        except Exception as e:
            logger.warning(f"Failed to reset offset to end: {e}. Continuing with default offset behavior...")
            self.last_partition_assignment = set()
            # Vẫn lưu timestamp để kiểm tra
            self.service_start_time = datetime.now()

        try:
            for message in self.consumer:
                if not self.is_running:
                    break
                
                # Kiểm tra nếu partition assignment thay đổi (rebalance)
                # Nếu có, seek to end và cập nhật service_start_time để chỉ đọc message mới
                current_assignment = self.consumer.assignment()
                if current_assignment != self.last_partition_assignment:
                    if current_assignment:
                        logger.info(f"Partition assignment changed (rebalance detected). Seeking to end for {len(current_assignment)} partitions...")
                        try:
                            # Cập nhật service_start_time khi rebalance để chỉ xử lý message mới
                            self.service_start_time = datetime.now()
                            # Reset first_message_timestamp để bắt đầu batch mới
                            self.first_message_timestamp = None
                            logger.info(f"Updated service ready time after rebalance: {self.service_start_time.isoformat()}, reset first_message_timestamp")
                            self.consumer.seek_to_end(*current_assignment)
                            logger.info(f"Seeked to end after rebalance - will only process new messages")
                        except Exception as e:
                            logger.warning(f"Failed to seek to end after rebalance: {e}")
                    self.last_partition_assignment = current_assignment.copy() if current_assignment else set()

                try:
                    # Lấy dữ liệu từ message
                    raw_data = message.value
                    original_key = message.key
                    
                    # Kiểm tra timestamp của message để chỉ xử lý message mới
                    # Bỏ qua message có timestamp trước khi service khởi động
                    # Điều này đảm bảo không xử lý message cũ ngay cả khi rebalance
                    if isinstance(raw_data, dict) and 'timestamp' in raw_data and self.service_start_time:
                        try:
                            msg_ts_str = str(raw_data['timestamp'])
                            # Parse ISO format timestamp
                            if 'T' in msg_ts_str:
                                # Parse timestamp để so sánh chính xác
                                # Remove timezone và parse
                                msg_ts_clean = msg_ts_str.split('+')[0].split('Z')[0]
                                try:
                                    # Parse ISO format với microseconds
                                    if '.' in msg_ts_clean:
                                        msg_dt = datetime.fromisoformat(msg_ts_clean)
                                    else:
                                        msg_dt = datetime.fromisoformat(msg_ts_clean)
                                    
                                    # So sánh với service start time (trừ 5 giây buffer để tránh timing issues)
                                    # Chỉ xử lý message có timestamp >= (service_start_time - 5s)
                                    buffer_time = self.service_start_time - timedelta(seconds=5)
                                    
                                    if msg_dt < buffer_time:
                                        logger.info(f"SKIPPING old message - msg_ts: {raw_data['timestamp']}, service_ready: {self.service_start_time.isoformat()}, diff: {(buffer_time - msg_dt).total_seconds():.2f}s")
                                        # Vẫn commit offset để không đọc lại
                                        try:
                                            self.consumer.commit()
                                        except:
                                            pass
                                        continue
                                    else:
                                        # Nếu đây là message đầu tiên được xử lý, lưu timestamp của nó
                                        # Từ đó chỉ xử lý message có timestamp >= message đầu tiên
                                        if self.first_message_timestamp is None:
                                            self.first_message_timestamp = msg_dt
                                            logger.info(f"First message timestamp set: {self.first_message_timestamp.isoformat()}")
                                        # Nếu đã có first_message_timestamp, chỉ xử lý message >= nó
                                        elif msg_dt < self.first_message_timestamp:
                                            logger.info(f"SKIPPING message before first message - msg_ts: {raw_data['timestamp']}, first_msg_ts: {self.first_message_timestamp.isoformat()}")
                                            try:
                                                self.consumer.commit()
                                            except:
                                                pass
                                            continue
                                        logger.debug(f"Processing new message - msg_ts: {raw_data['timestamp']}, service_ready: {self.service_start_time.isoformat()}")
                                except ValueError:
                                    # Nếu không parse được, xử lý bình thường
                                    logger.debug(f"Could not parse message timestamp format: {msg_ts_clean}, processing anyway")
                        except Exception as e:
                            logger.debug(f"Could not compare message timestamp: {e}, processing anyway")

                    logger.info(f"Processing record with key: {original_key}")

                    # Tiền xử lý dữ liệu
                    processed_data = self.preprocess_single_record(raw_data)

                    # Cập nhật summary statistics
                    self._update_summary(processed_data)
                    
                    # Gửi dữ liệu đã xử lý
                    self.send_processed_data(processed_data, original_key)
                    
                    # Commit offset thủ công sau mỗi message để đảm bảo không đọc lại
                    # Điều này đảm bảo khi chạy simulate_attack_service.py nhiều lần,
                    # consumer chỉ đọc message mới, không duplicate
                    try:
                        self.consumer.commit()
                    except Exception as e:
                        logger.warning(f"Failed to commit offset: {e}")
                    
                    # Log summary mỗi 10 records
                    if self.processed_count % 10 == 0:
                        self._log_summary()

                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    self.error_count += 1
                    continue

        except KeyboardInterrupt:
            logger.info("Preprocessing service stopped by user")
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
        finally:
            self.stop()

    def _update_summary(self, processed_record: Dict[str, Any]):
        """Cập nhật summary statistics"""
        self.processed_count += 1
        
        # Lấy label và label_group từ record
        label = processed_record.get('label', 'unknown')
        label_group = processed_record.get('label_group', 'unknown')
        
        if label not in self.label_summary:
            self.label_summary[label] = 0
        self.label_summary[label] += 1
        
        if label_group not in self.label_group_summary:
            self.label_group_summary[label_group] = 0
        self.label_group_summary[label_group] += 1
    
    def _log_summary(self):
        """Log summary statistics"""
        logger.info("=" * 60)
        logger.info("DATA PREPROCESSING SUMMARY:")
        logger.info("=" * 60)
        logger.info(f"Total records processed: {self.processed_count}")
        logger.info(f"Errors: {self.error_count}")
        logger.info("")
        logger.info("Label distribution:")
        for label, count in sorted(self.label_summary.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / self.processed_count) * 100 if self.processed_count > 0 else 0
            logger.info(f"  - {label}: {count} ({percentage:.1f}%)")
        logger.info("")
        logger.info("Label group distribution:")
        for group, count in sorted(self.label_group_summary.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / self.processed_count) * 100 if self.processed_count > 0 else 0
            logger.info(f"  - {group}: {count} ({percentage:.1f}%)")
        logger.info("=" * 60)
    
    def stop(self):
        """Dừng service"""
        self.is_running = False
        
        # Log final summary
        if self.processed_count > 0:
            logger.info("")
            self._log_summary()
        
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()
            logger.info("Data preprocessing service stopped")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Safenet IDS - Data Preprocessing Service')
    parser.add_argument('--kafka-servers', default='localhost:9092',
                       help='Kafka bootstrap servers')
    parser.add_argument('--input-topic', default='raw_data_event',
                       help='Input topic name')
    parser.add_argument('--output-topic', default='preprocess_data',
                       help='Output topic name')
    parser.add_argument('--model-type', choices=('random_forest', 'cnn_lstm'),
                       default='random_forest',
                       help='Loại model sử dụng: random_forest hoặc cnn_lstm')

    args = parser.parse_args()

    # Tạo thư mục logs nếu chưa có
    import os
    os.makedirs('services/logs', exist_ok=True)

    # Khởi tạo và chạy service
    service = DataPreprocessingService(
        kafka_bootstrap_servers=args.kafka_servers,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        model_type=args.model_type
    )

    try:
        logger.info("Starting Safenet IDS Data Preprocessing Service...")
        service.start_processing()
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service failed: {e}")


if __name__ == '__main__':
    main()
