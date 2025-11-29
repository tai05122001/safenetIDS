#!/usr/bin/env python3
"""
Safenet IDS - Data Preprocessing Service
Dịch vụ tiền xử lý dữ liệu network từ Kafka topic 'raw_network_events'

Chức năng chính:
- Đọc dữ liệu thô từ Kafka raw_network_events
- Áp dụng các bước preprocessing giống như trong training pipeline
- Chuẩn hóa tên cột, xử lý missing values, scaling, encoding
- Gửi dữ liệu đã xử lý đến topic preprocessed_events

Luồng xử lý:
1. Nhận message từ raw_network_events
2. Parse JSON thành Python dict
3. Áp dụng preprocessing pipeline
4. Gửi kết quả đến preprocessed_events
5. Log và monitoring
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
from datetime import datetime  # Timestamp handling

# Cấu hình logging system
# - Level INFO: Ghi các event quan trọng
# - Format: timestamp, logger name, level, message
# - Handlers: File log và console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('services/logs/data_preprocessing.log'),  # Persistent log file
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

    Pipeline preprocessing (giống training):
    1. Normalize column names
    2. Convert numeric columns
    3. Fill missing values
    4. Add label_group column
    5. Drop sparse/constant columns
    6. IQR outlier clipping
    7. Standard scaling
    """

    def __init__(self,
                 kafka_bootstrap_servers='localhost:9092',
                 input_topic='raw_data_event',
                 output_topic='preprocess_data'):
        """
        Khởi tạo Data Preprocessing Service

        Args:
            kafka_bootstrap_servers: Kafka bootstrap servers
            input_topic: Topic để đọc dữ liệu thô
            output_topic: Topic để gửi dữ liệu đã xử lý
        """
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.consumer = None
        self.producer = None
        self.kafka_servers = kafka_bootstrap_servers
        self.is_running = False

        # Metadata để tái sử dụng logic preprocessing
        self.preprocessing_metadata = {}

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
                auto_offset_reset='earliest',
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
        Chuẩn hóa tên cột để đảm bảo consistency

        Process:
        1. Convert to lowercase (đảm bảo uniformity)
        2. Replace special characters bằng underscore (regex [^\w\s] -> _)
        3. Replace multiple spaces bằng single underscore (regex \s+ -> _)

        Examples:
        - "Flow Duration" -> "flow_duration"
        - "Total Fwd Packets" -> "total_fwd_packets"
        - "Fwd Packet Length Max" -> "fwd_packet_length_max"

        Args:
            name (str): Tên cột gốc

        Returns:
            str: Tên cột đã chuẩn hóa
        """
        # Bước 1: Convert to lowercase
        name = name.lower()

        # Bước 2: Replace ký tự đặc biệt (không phải word char hoặc space) bằng _
        # \w = word characters (letters, digits, underscore), \s = whitespace
        # [^\w\s] = NOT word chars AND NOT whitespace -> special chars
        name = re.sub(r'[^\w\s]', '_', name)

        # Bước 3: Replace multiple spaces bằng single underscore
        # \s+ = one or more whitespace characters
        name = re.sub(r'\s+', '_', name)

        return name

    @staticmethod
    def convert_numeric(df: pd.DataFrame, skip_cols: Iterable[str]) -> pd.DataFrame:
        """
        Chuyển đổi các cột về đúng data type numeric

        Process:
        - Duyệt qua tất cả columns (trừ skip_cols)
        - Sử dụng pd.to_numeric() với errors='coerce'
        - 'coerce' sẽ convert giá trị invalid thành NaN thay vì raise error

        Skip columns thường bao gồm:
        - timestamp, source_ip, destination_ip (string identifiers)
        - label, label_group (categorical target variables)

        Args:
            df (pd.DataFrame): DataFrame đầu vào
            skip_cols (Iterable[str]): Danh sách cột không convert

        Returns:
            pd.DataFrame: DataFrame với các cột đã convert sang numeric
        """
        # Tạo copy để không modify DataFrame gốc
        df_copy = df.copy()

        # Duyệt qua từng cột
        for col in df_copy.columns:
            # Bỏ qua các cột được chỉ định (identifiers, categorical)
            if col not in skip_cols:
                try:
                    # Thử convert sang numeric
                    # errors='coerce': Invalid values -> NaN (không raise exception)
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

                except Exception as e:
                    # Log warning nếu convert thất bại (rare case)
                    logger.warning(f"Could not convert column {col} to numeric: {e}")

        return df_copy

    @staticmethod
    def fill_missing_values(df: pd.DataFrame, skip_cols: Iterable[str]) -> pd.DataFrame:
        """
        Điền các giá trị thiếu (NaN) trong DataFrame

        Chiến lược imputation:
        - Numeric columns: Sử dụng median (robust hơn mean với outliers)
        - Categorical columns: Sử dụng mode (most frequent value)

        Lý do dùng median cho numeric:
        - Ít bị ảnh hưởng bởi outliers
        - Giữ được distribution shape
        - Phù hợp với network traffic data có nhiều outliers

        Args:
            df (pd.DataFrame): DataFrame có thể chứa NaN
            skip_cols (Iterable[str]): Cột không cần imputation

        Returns:
            pd.DataFrame: DataFrame đã được fill missing values
        """
        df_copy = df.copy()

        for col in df_copy.columns:
            if col not in skip_cols:
                # Kiểm tra data type của cột
                if df_copy[col].dtype in ['int64', 'float64']:
                    # Numeric columns: dùng median
                    median_val = df_copy[col].median()
                    if pd.notna(median_val):  # Đảm bảo median không NaN
                        df_copy[col] = df_copy[col].fillna(median_val)
                else:
                    # Categorical columns: dùng mode (most frequent)
                    mode_val = df_copy[col].mode()
                    if not mode_val.empty:  # Đảm bảo có mode
                        df_copy[col] = df_copy[col].fillna(mode_val[0])

        return df_copy

    @staticmethod
    def add_label_group_column(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Tạo cột label_group từ label gốc (giả lập cho real-time processing)
        Trong production, logic này sẽ được điều chỉnh theo nhu cầu
        """
        df_copy = df.copy()

        # Mapping từ label gốc sang label_group (dựa trên CICIDS2017)
        label_to_group = {
            'BENIGN': 'benign',
            'DoS Hulk': 'dos',
            'DoS GoldenEye': 'dos',
            'DoS slowloris': 'dos',
            'DoS Slowhttptest': 'dos',
            'DoS attack': 'dos',
            'Heartbleed': 'dos',
            'DDoS': 'ddos',
            'DDOS attack-HOIC': 'ddos',
            'DDOS attack-LOIC-UDP': 'ddos',
            'Bot': 'bot',
            'Web Attack': 'rare_attack',
            'Brute Force': 'rare_attack',
            'XSS': 'rare_attack',
            'SQL Injection': 'rare_attack',
            'Infiltration': 'rare_attack',
            'FTP-Patator': 'rare_attack',
            'SSH-Patator': 'rare_attack'
        }

        # Normalize label column
        if label_col in df_copy.columns:
            df_copy[label_col] = df_copy[label_col].str.strip().str.title()
            df_copy['label_group'] = df_copy[label_col].map(label_to_group).fillna('benign')
        else:
            # Nếu không có label (real-time data), mặc định là benign
            df_copy['label_group'] = 'benign'

        return df_copy, label_to_group

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
    def drop_constant_columns(df: pd.DataFrame, skip_cols: Iterable[str]) -> pd.DataFrame:
        """
        Loại bỏ các cột chỉ có một giá trị duy nhất
        """
        df_copy = df.copy()
        cols_to_drop = []

        for col in df_copy.columns:
            if col not in skip_cols:
                if df_copy[col].nunique() == 1:
                    cols_to_drop.append(col)
                    logger.info(f"Dropping constant column {col}")

        df_copy = df_copy.drop(columns=cols_to_drop)
        return df_copy

    @staticmethod
    def clip_outliers_iqr(df: pd.DataFrame, iqr_factor: float, skip_cols: Iterable[str]) -> pd.DataFrame:
        """
        Clip outliers theo phương pháp IQR
        """
        df_copy = df.copy()

        for col in df_copy.columns:
            if col not in skip_cols and df_copy[col].dtype in ['int64', 'float64']:
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_factor * IQR
                upper_bound = Q3 + iqr_factor * IQR

                # Clip values
                df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)

        return df_copy

    @staticmethod
    def scale_numeric_features(df: pd.DataFrame, method: str, skip_cols: Iterable[str]) -> Tuple[pd.DataFrame, Dict]:
        """
        Chuẩn hóa các đặc trưng numeric
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        df_copy = df.copy()
        scaler_info = {}

        numeric_cols = [col for col in df_copy.columns
                       if col not in skip_cols and df_copy[col].dtype in ['int64', 'float64']]

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            return df_copy, scaler_info

        if numeric_cols:
            df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])
            scaler_info = {
                'method': method,
                'columns': numeric_cols,
                'scaler_params': {
                    'mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
                    'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
                    'min': scaler.min_.tolist() if hasattr(scaler, 'min_') else None,
                    'scale_minmax': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None
                }
            }

        return df_copy, scaler_info

    def preprocess_single_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tiền xử lý một record đơn lẻ

        Args:
            record: Dictionary chứa dữ liệu network thô

        Returns:
            Dictionary chứa dữ liệu đã tiền xử lý
        """
        try:
            # Chuyển thành DataFrame để xử lý
            df = pd.DataFrame([record])

            # Normalize column names
            df.columns = [self.normalize_column(col) for col in df.columns]

            # Skip columns
            skip_cols = ['timestamp', 'source_ip', 'destination_ip', 'label', 'label_group']

            # Convert numeric
            df = self.convert_numeric(df, skip_cols)

            # Fill missing values
            df = self.fill_missing_values(df, skip_cols)

            # Add label group column
            df, label_mapping = self.add_label_group_column(df, 'label')

            # Drop sparse columns (skip nếu chỉ có 1 record)
            # df = self.drop_sparse_columns(df, 0.6, skip_cols)

            # Drop constant columns
            df = self.drop_constant_columns(df, skip_cols)

            # Clip outliers
            df = self.clip_outliers_iqr(df, 1.5, skip_cols)

            # Scale numeric features (standard scaling)
            df, scaler_info = self.scale_numeric_features(df, 'standard', skip_cols)

            # Chuyển về dictionary
            processed_record = df.iloc[0].to_dict()

            # Thêm metadata preprocessing
            processed_record['preprocessing_timestamp'] = datetime.now().isoformat()
            processed_record['preprocessing_metadata'] = {
                'label_mapping': label_mapping,
                'scaler_info': scaler_info,
                'processed_columns': list(df.columns)
            }

            return processed_record

        except Exception as e:
            logger.error(f"Error preprocessing record: {e}")
            # Trả về record gốc với flag error
            record['preprocessing_error'] = str(e)
            record['preprocessing_timestamp'] = datetime.now().isoformat()
            return record

    def send_processed_data(self, data: Dict[str, Any], original_key: str = None):
        """
        Gửi dữ liệu đã xử lý đến Kafka topic

        Args:
            data: Dữ liệu đã tiền xử lý
            original_key: Key từ message gốc
        """
        try:
            key = original_key or data.get('timestamp', str(datetime.now().timestamp()))

            future = self.producer.send(self.output_topic, value=data, key=key)
            record_metadata = future.get(timeout=10)

            logger.info(f"Sent processed data to {record_metadata.topic} "
                       f"partition {record_metadata.partition} "
                       f"offset {record_metadata.offset}")

        except Exception as e:
            logger.error(f"Failed to send processed data: {e}")

    def start_processing(self):
        """Bắt đầu quá trình xử lý dữ liệu"""
        self.is_running = True
        logger.info(f"Starting data preprocessing service: {self.input_topic} -> {self.output_topic}")

        try:
            for message in self.consumer:
                if not self.is_running:
                    break

                try:
                    # Lấy dữ liệu từ message
                    raw_data = message.value
                    original_key = message.key

                    logger.info(f"Processing record with key: {original_key}")

                    # Tiền xử lý dữ liệu
                    processed_data = self.preprocess_single_record(raw_data)

                    # Gửi dữ liệu đã xử lý
                    self.send_processed_data(processed_data, original_key)

                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue

        except KeyboardInterrupt:
            logger.info("Preprocessing service stopped by user")
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
        finally:
            self.stop()

    def stop(self):
        """Dừng service"""
        self.is_running = False
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

    args = parser.parse_args()

    # Tạo thư mục logs nếu chưa có
    import os
    os.makedirs('services/logs', exist_ok=True)

    # Khởi tạo và chạy service
    service = DataPreprocessingService(
        kafka_bootstrap_servers=args.kafka_servers,
        input_topic=args.input_topic,
        output_topic=args.output_topic
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
