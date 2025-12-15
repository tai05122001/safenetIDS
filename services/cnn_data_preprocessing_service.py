#!/usr/bin/env python3
"""
CNN Data Preprocessing Service
Xử lý dữ liệu cho CNN models với preprocessing phù hợp cho 1D CNN
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from datetime import datetime, timedelta

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Cấu hình logging
os.makedirs('services/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('services/logs/cnn_data_preprocessing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CNNDataPreprocessing')

class CNNDataPreprocessingService:
    """Service xử lý dữ liệu cho CNN models"""

    def __init__(self,
                 kafka_bootstrap_servers='localhost:9092',
                 input_topic='raw_data_event_cnn',
                 output_topic='preprocess_data_cnn',
                 group_id='safenet-cnn-preprocessing-group'):
        """
        Khởi tạo CNN Data Preprocessing Service

        Args:
            kafka_bootstrap_servers: Kafka bootstrap servers
            input_topic: Topic để đọc raw data
            output_topic: Topic để gửi processed data cho CNN
            group_id: Consumer group ID
        """
        self.kafka_servers = kafka_bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.group_id = group_id

        # Kafka clients
        self.consumer = None
        self.producer = None

        # Scalers cho CNN (nếu cần)
        self.scaler = None
        self.label_encoder = None

        # Feature columns từ training data - CẦN ĐỂ ĐẢM BẢO THỨ TỰ ĐÚNG
        self.feature_columns = None
        self.raw_to_processed_mapping = None

        # Stats
        self.processed_count = 0
        self.error_count = 0
        self.valid_records_count = 0
        self.invalid_records_count = 0
        self.start_time = datetime.now()

        # Summary statistics (giống data_preprocessing_service.py)
        self.label_group_summary = {}
        self.label_summary = {}

        # Attack/Benign counters for summary
        self.benign_count = 0
        self.attack_count = 0
        self.dos_count = 0
        self.ddos_count = 0
        self.portscan_count = 0

        # Timestamp khi service khởi động để chỉ xử lý message mới
        self.service_start_time = None
        # Lưu partition assignment để detect rebalance
        self.last_partition_assignment = set()
        # Lưu timestamp của message đầu tiên được xử lý
        # Điều này đảm bảo chỉ xử lý message có timestamp >= message đầu tiên
        self.first_message_timestamp = None

        # Khởi tạo
        self._load_feature_columns_from_training()
        self._validate_feature_consistency()  # Validate feature order consistency
        self._init_scaler()
        self._init_kafka()

    def _validate_feature_consistency(self):
        """Validate that feature order is consistent between metadata and hardcoded fallback"""
        try:
            metadata_path = Path("artifacts_cnn/training_metadata.json")
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                metadata_features = metadata.get('data_info', {}).get('feature_columns', [])
                hardcoded_features = self._get_hardcoded_feature_columns()

                if metadata_features and metadata_features != hardcoded_features:
                    logger.error("CRITICAL: Feature order mismatch!")
                    logger.error(f"Metadata features ({len(metadata_features)}): {metadata_features}")
                    logger.error(f"Hardcoded features ({len(hardcoded_features)}): {hardcoded_features}")

                    # Find differences
                    missing_in_metadata = set(hardcoded_features) - set(metadata_features)
                    extra_in_metadata = set(metadata_features) - set(hardcoded_features)

                    if missing_in_metadata:
                        logger.error(f"Features missing in metadata: {missing_in_metadata}")
                    if extra_in_metadata:
                        logger.error(f"Extra features in metadata: {extra_in_metadata}")

                    # This is critical - should stop service
                    raise ValueError("Feature order inconsistency detected. Retrain model or fix hardcoded fallback.")
                else:
                    logger.info("Feature order consistency validated ✓")
        except Exception as e:
            logger.error(f"Feature consistency validation failed: {e}")
            raise

    def _load_feature_columns_from_training(self):
        """Load feature columns từ training metadata để đảm bảo thứ tự đúng"""
        try:
            metadata_path = Path("artifacts_cnn/training_metadata.json")
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                self.feature_columns = metadata.get('data_info', {}).get('feature_columns', [])
                if self.feature_columns:
                    logger.info(f"Loaded {len(self.feature_columns)} feature columns from training metadata")
                    logger.info(f"First 5 features: {self.feature_columns[:5]}")
                    logger.info(f"Last 5 features: {self.feature_columns[-5:]}")

                    # Tạo mapping từ processed names sang raw names
                    self.processed_to_raw_mapping = self._create_processed_to_raw_mapping()

                    # Verify consistency with hardcoded fallback
                    hardcoded_features = self._get_hardcoded_feature_columns()
                    if self.feature_columns != hardcoded_features:
                        logger.warning("Feature columns from metadata differ from hardcoded fallback!")
                        logger.warning(f"Metadata: {len(self.feature_columns)} features")
                        logger.warning(f"Hardcoded: {len(hardcoded_features)} features")
                        # Use metadata but log warning
                    else:
                        logger.info("Feature columns consistent between metadata and hardcoded fallback")
                else:
                    logger.warning("No feature columns found in metadata, using hardcoded fallback order")
                    self.feature_columns = self._get_hardcoded_feature_columns()
                    self.processed_to_raw_mapping = self._create_processed_to_raw_mapping()
            else:
                logger.warning("Training metadata not found, using hardcoded fallback feature order")
                self.feature_columns = self._get_hardcoded_feature_columns()
                self.processed_to_raw_mapping = self._create_processed_to_raw_mapping()

        except Exception as e:
            logger.error(f"Failed to load feature columns from training: {e}")
            logger.warning("Using hardcoded fallback feature order to ensure compatibility")
            self.feature_columns = self._get_hardcoded_feature_columns()
            self.processed_to_raw_mapping = self._create_processed_to_raw_mapping()

    def _get_hardcoded_feature_columns(self) -> list:
        """
        Hardcoded feature columns in the exact order used during training
        This ensures compatibility even when training_metadata.json cannot be loaded
        """
        return [
            "bwd_avg_bulk_rate",
            "avg_bwd_segment_size",
            "flow_iat_std",
            "total_length_of_bwd_packets",
            "bwd_iat_min",
            "idle_min",
            "total_length_of_fwd_packets",
            "fwd_iat_total",
            "packet_length_mean",
            "bwd_packet_length_mean",
            "bwd_iat_std",
            "destination_port",
            "bwd_avg_packets_bulk",
            "fwd_urg_flags",
            "bwd_urg_flags",
            "cwe_flag_count",
            "flow_iat_max",
            "active_std",
            "bwd_packet_length_min",
            "active_min",
            "fwd_packet_length_max",
            "bwd_packet_length_max",
            "flow_packets_s",
            "bwd_header_length",
            "psh_flag_count",
            "average_packet_size",
            "init_win_bytes_backward",
            "fwd_psh_flags",
            "init_win_bytes_forward",
            "subflow_fwd_packets",
            "fwd_iat_min",
            "total_backward_packets",
            "packet_length_variance",
            "fin_flag_count",
            "fwd_packet_length_mean",
            "bwd_avg_bytes_bulk",
            "packet_length_std",
            "ece_flag_count",
            "urg_flag_count",
            "fwd_packet_length_std",
            "max_packet_length",
            "min_packet_length",
            "active_mean",
            "fwd_iat_std",
            "fwd_iat_max",
            "bwd_packets_s",
            "avg_fwd_segment_size",
            "bwd_iat_max",
            "flow_iat_mean",
            "min_seg_size_forward",
            "idle_mean",
            "subflow_bwd_packets",
            "bwd_iat_mean",
            "flow_duration",
            "bwd_psh_flags",
            "fwd_header_length_1",
            "idle_max",
            "bwd_iat_total",
            "flow_bytes_s",
            "bwd_packet_length_std",
            "fwd_avg_packets_bulk",
            "rst_flag_count",
            "act_data_pkt_fwd",
            "active_max",
            "flow_iat_min",
            "fwd_packet_length_min",
            "idle_std",
            "fwd_iat_mean",
            "subflow_fwd_bytes",
            "subflow_bwd_bytes",
            "fwd_avg_bytes_bulk",
            "total_fwd_packets",
            "down_up_ratio",
            "fwd_packets_s",
            "syn_flag_count",
            "fwd_avg_bulk_rate",
            "ack_flag_count",
            "fwd_header_length"
        ]

    def _create_processed_to_raw_mapping(self) -> Dict[str, str]:
        """Tạo mapping từ processed feature names sang raw names"""
        # Mapping từ processed names (snake_case) sang raw CICIDS2017 names (with spaces)
        mapping = {
            'destination_port': ' Destination Port',
            'flow_duration': ' Flow Duration',
            'total_fwd_packets': ' Total Fwd Packets',
            'total_backward_packets': ' Total Backward Packets',
            'total_length_of_fwd_packets': 'Total Length of Fwd Packets',
            'total_length_of_bwd_packets': ' Total Length of Bwd Packets',
            'fwd_packet_length_max': ' Fwd Packet Length Max',
            'fwd_packet_length_min': ' Fwd Packet Length Min',
            'fwd_packet_length_mean': ' Fwd Packet Length Mean',
            'fwd_packet_length_std': ' Fwd Packet Length Std',
            'bwd_packet_length_max': 'Bwd Packet Length Max',
            'bwd_packet_length_min': ' Bwd Packet Length Min',
            'bwd_packet_length_mean': ' Bwd Packet Length Mean',
            'bwd_packet_length_std': ' Bwd Packet Length Std',
            'flow_bytes_s': 'Flow Bytes/s',
            'flow_packets_s': ' Flow Packets/s',
            'flow_iat_mean': ' Flow IAT Mean',
            'flow_iat_std': ' Flow IAT Std',
            'flow_iat_max': ' Flow IAT Max',
            'flow_iat_min': ' Flow IAT Min',
            'fwd_iat_total': 'Fwd IAT Total',
            'fwd_iat_mean': ' Fwd IAT Mean',
            'fwd_iat_std': ' Fwd IAT Std',
            'fwd_iat_max': ' Fwd IAT Max',
            'fwd_iat_min': ' Fwd IAT Min',
            'bwd_iat_total': 'Bwd IAT Total',
            'bwd_iat_mean': ' Bwd IAT Mean',
            'bwd_iat_std': ' Bwd IAT Std',
            'bwd_iat_max': ' Bwd IAT Max',
            'bwd_iat_min': ' Bwd IAT Min',
            'fwd_psh_flags': 'Fwd PSH Flags',
            'bwd_psh_flags': ' Bwd PSH Flags',
            'fwd_urg_flags': ' Fwd URG Flags',
            'bwd_urg_flags': ' Bwd URG Flags',
            'fwd_header_length': ' Fwd Header Length',
            'bwd_header_length': ' Bwd Header Length',
            'fwd_packets_s': 'Fwd Packets/s',
            'bwd_packets_s': ' Bwd Packets/s',
            'min_packet_length': ' Min Packet Length',
            'max_packet_length': ' Max Packet Length',
            'packet_length_mean': ' Packet Length Mean',
            'packet_length_std': ' Packet Length Std',
            'packet_length_variance': ' Packet Length Variance',
            'fin_flag_count': 'FIN Flag Count',
            'syn_flag_count': ' SYN Flag Count',
            'rst_flag_count': ' RST Flag Count',
            'psh_flag_count': ' PSH Flag Count',
            'ack_flag_count': ' ACK Flag Count',
            'urg_flag_count': ' URG Flag Count',
            'cwe_flag_count': ' CWE Flag Count',
            'ece_flag_count': ' ECE Flag Count',
            'down_up_ratio': ' Down/Up Ratio',
            'average_packet_size': ' Average Packet Size',
            'avg_fwd_segment_size': ' Avg Fwd Segment Size',
            'avg_bwd_segment_size': ' Avg Bwd Segment Size',
            'fwd_header_length_1': ' Fwd Header Length.1',
            'fwd_avg_bytes_bulk': 'Fwd Avg Bytes/Bulk',
            'fwd_avg_packets_bulk': ' Fwd Avg Packets/Bulk',
            'fwd_avg_bulk_rate': ' Fwd Avg Bulk Rate',
            'bwd_avg_bytes_bulk': ' Bwd Avg Bytes/Bulk',
            'bwd_avg_packets_bulk': ' Bwd Avg Packets/Bulk',
            'bwd_avg_bulk_rate': 'Bwd Avg Bulk Rate',
            'subflow_fwd_packets': 'Subflow Fwd Packets',
            'subflow_fwd_bytes': ' Subflow Fwd Bytes',
            'subflow_bwd_packets': ' Subflow Bwd Packets',
            'subflow_bwd_bytes': ' Subflow Bwd Bytes',
            'init_win_bytes_forward': 'Init_Win_bytes_forward',
            'init_win_bytes_backward': ' Init_Win_bytes_backward',
            'act_data_pkt_fwd': ' act_data_pkt_fwd',
            'min_seg_size_forward': ' min_seg_size_forward',
            'active_mean': 'Active Mean',
            'active_std': ' Active Std',
            'active_max': ' Active Max',
            'active_min': ' Active Min',
            'idle_mean': 'Idle Mean',
            'idle_std': ' Idle Std',
            'idle_max': ' Idle Max',
            'idle_min': ' Idle Min'
        }
        return mapping

    def _init_scaler(self):
        """Khởi tạo scaler cho CNN preprocessing - BẮT BUỘC phải load scaler đã trained"""
        try:
            # Load scaler từ Level 1 CNN artifacts (BẮT BUỘC)
            scaler_path = Path("artifacts_cnn/scaler.joblib")
            if not scaler_path.exists():
                error_msg = f"❌ CRITICAL: Scaler file not found at {scaler_path}! Cannot proceed without trained scaler."
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            self.scaler = joblib.load(scaler_path)
            logger.info("✅ Loaded scaler from Level 1 CNN artifacts")
            
            # Verify scaler is fitted
            if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
                error_msg = "❌ CRITICAL: Scaler loaded but NOT fitted! Cannot use unfitted scaler."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"✅ Scaler is fitted: {len(self.scaler.mean_)} features")
            logger.info(f"   Scaler mean range: [{self.scaler.mean_.min():.2f}, {self.scaler.mean_.max():.2f}]")
            logger.info(f"   Scaler scale range: [{self.scaler.scale_.min():.2f}, {self.scaler.scale_.max():.2f}]")

            # Load label encoder từ Level 1
            le_path = Path("artifacts_cnn/label_encoder.joblib")
            if le_path.exists():
                self.label_encoder = joblib.load(le_path)
                logger.info("✅ Loaded label encoder from Level 1 CNN artifacts")
            else:
                # Label encoder không quan trọng bằng scaler, có thể tạo mới
                self.label_encoder = LabelEncoder()
                logger.warning("⚠️  Label encoder not found, created new one (not critical)")

        except Exception as e:
            logger.error(f"❌ FATAL: Failed to initialize scaler: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise  # Dừng service nếu không load được scaler

    def _init_kafka(self):
        """Khởi tạo Kafka consumer và producer"""
        try:
            # Consumer với cấu hình giống data_preprocessing_service.py
            self.consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.kafka_servers,
                group_id=self.group_id,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                key_deserializer=lambda x: x.decode('utf-8') if x else None,
                auto_offset_reset='earliest',  # Đọc toàn bộ backlog để không bỏ lỡ message
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                session_timeout_ms=30000,
                heartbeat_interval_ms=3000,
                max_poll_records=100
            )
            logger.info(f"Kafka consumer initialized for topic: {self.input_topic}")

            # Producer
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
            logger.error(f"Failed to initialize Kafka: {e}")
            raise

    def _load_feature_mapping(self) -> Dict[str, str]:
        """
        Load mapping từ raw feature names sang processed feature names
        Sử dụng cùng mapping như trong training data
        """
        # Mapping từ raw CICIDS2017 feature names sang processed names
        # (loại bỏ space prefix và chuyển thành snake_case)
        feature_mapping = {
            # Original CICIDS2017 names -> Processed names (snake_case, no spaces)
            ' Destination Port': 'destination_port',
            ' Flow Duration': 'flow_duration',
            ' Total Fwd Packets': 'total_fwd_packets',
            ' Total Backward Packets': 'total_backward_packets',
            'Total Length of Fwd Packets': 'total_length_of_fwd_packets',
            ' Total Length of Bwd Packets': 'total_length_of_bwd_packets',
            ' Fwd Packet Length Max': 'fwd_packet_length_max',
            ' Fwd Packet Length Min': 'fwd_packet_length_min',
            ' Fwd Packet Length Mean': 'fwd_packet_length_mean',
            ' Fwd Packet Length Std': 'fwd_packet_length_std',
            'Bwd Packet Length Max': 'bwd_packet_length_max',
            ' Bwd Packet Length Min': 'bwd_packet_length_min',
            ' Bwd Packet Length Mean': 'bwd_packet_length_mean',
            ' Bwd Packet Length Std': 'bwd_packet_length_std',
            'Flow Bytes/s': 'flow_bytes_s',
            ' Flow Packets/s': 'flow_packets_s',
            ' Flow IAT Mean': 'flow_iat_mean',
            ' Flow IAT Std': 'flow_iat_std',
            ' Flow IAT Max': 'flow_iat_max',
            ' Flow IAT Min': 'flow_iat_min',
            'Fwd IAT Total': 'fwd_iat_total',
            ' Fwd IAT Mean': 'fwd_iat_mean',
            ' Fwd IAT Std': 'fwd_iat_std',
            ' Fwd IAT Max': 'fwd_iat_max',
            ' Fwd IAT Min': 'fwd_iat_min',
            'Bwd IAT Total': 'bwd_iat_total',
            ' Bwd IAT Mean': 'bwd_iat_mean',
            ' Bwd IAT Std': 'bwd_iat_std',
            ' Bwd IAT Max': 'bwd_iat_max',
            ' Bwd IAT Min': 'bwd_iat_min',
            'Fwd PSH Flags': 'fwd_psh_flags',
            ' Bwd PSH Flags': 'bwd_psh_flags',
            ' Fwd URG Flags': 'fwd_urg_flags',
            ' Bwd URG Flags': 'bwd_urg_flags',
            ' Fwd Header Length': 'fwd_header_length',
            ' Bwd Header Length': 'bwd_header_length',
            'Fwd Packets/s': 'fwd_packets_s',
            ' Bwd Packets/s': 'bwd_packets_s',
            ' Min Packet Length': 'min_packet_length',
            ' Max Packet Length': 'max_packet_length',
            ' Packet Length Mean': 'packet_length_mean',
            ' Packet Length Std': 'packet_length_std',
            ' Packet Length Variance': 'packet_length_variance',
            'FIN Flag Count': 'fin_flag_count',
            ' SYN Flag Count': 'syn_flag_count',
            ' RST Flag Count': 'rst_flag_count',
            ' PSH Flag Count': 'psh_flag_count',
            ' ACK Flag Count': 'ack_flag_count',
            ' URG Flag Count': 'urg_flag_count',
            ' CWE Flag Count': 'cwe_flag_count',
            ' ECE Flag Count': 'ece_flag_count',
            ' Down/Up Ratio': 'down_up_ratio',
            ' Average Packet Size': 'average_packet_size',
            ' Avg Fwd Segment Size': 'avg_fwd_segment_size',
            ' Avg Bwd Segment Size': 'avg_bwd_segment_size',
            ' Fwd Header Length.1': 'fwd_header_length_1',
            'Fwd Avg Bytes/Bulk': 'fwd_avg_bytes_bulk',
            ' Fwd Avg Packets/Bulk': 'fwd_avg_packets_bulk',
            ' Fwd Avg Bulk Rate': 'fwd_avg_bulk_rate',
            ' Bwd Avg Bytes/Bulk': 'bwd_avg_bytes_bulk',
            ' Bwd Avg Packets/Bulk': 'bwd_avg_packets_bulk',
            'Bwd Avg Bulk Rate': 'bwd_avg_bulk_rate',
            'Subflow Fwd Packets': 'subflow_fwd_packets',
            ' Subflow Fwd Bytes': 'subflow_fwd_bytes',
            ' Subflow Bwd Packets': 'subflow_bwd_packets',
            ' Subflow Bwd Bytes': 'subflow_bwd_bytes',
            'Init_Win_bytes_forward': 'init_win_bytes_forward',
            ' Init_Win_bytes_backward': 'init_win_bytes_backward',
            ' act_data_pkt_fwd': 'act_data_pkt_fwd',
            ' min_seg_size_forward': 'min_seg_size_forward',
            'Active Mean': 'active_mean',
            ' Active Std': 'active_std',
            ' Active Max': 'active_max',
            ' Active Min': 'active_min',
            'Idle Mean': 'idle_mean',
            ' Idle Std': 'idle_std',
            ' Idle Max': 'idle_max',
            ' Idle Min': 'idle_min'
        }
        return feature_mapping

    def _extract_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract numerical features từ raw data cho CNN
        Sử dụng đúng feature names và thứ tự như training data
        """
        features = {}
        found_count = 0
        missing_count = 0
        missing_features = []

        if self.feature_columns and self.processed_to_raw_mapping:
            # Sử dụng feature columns từ training metadata - đảm bảo thứ tự ĐÚNG
            for processed_name in self.feature_columns:
                # Thử nhiều cách để tìm feature:
                # 1. Tìm raw name (có spaces) - cho data từ simulate_attack_service
                # 2. Tìm processed name trực tiếp (snake_case) - cho data từ packet_capture_service
                # 3. Nếu không tìm thấy, set = 0
                raw_name = self.processed_to_raw_mapping.get(processed_name)
                value = None
                found_in_raw = False
                
                # Thử 1: Tìm raw format (có spaces) - cho data từ simulate_attack_service
                if raw_name and raw_name in data:
                    value = data[raw_name]
                    found_in_raw = True
                # Thử 2: Tìm processed format trực tiếp (snake_case) - cho data từ packet_capture_service
                elif processed_name in data:
                    value = data[processed_name]
                    found_in_raw = False
                
                if value is not None:
                    # Convert to float, handle inf/-inf/nan, also handle string numbers
                    try:
                        if isinstance(value, str):
                            # Handle string numbers like "80"
                            if value.strip() == '':
                                features[processed_name] = 0.0
                                missing_count += 1
                                missing_features.append(processed_name)
                            else:
                                numeric_value = float(value)
                                if np.isinf(numeric_value) or np.isnan(numeric_value):
                                    features[processed_name] = 0.0
                                    missing_count += 1
                                    missing_features.append(processed_name)
                                else:
                                    features[processed_name] = numeric_value
                                    found_count += 1
                        elif isinstance(value, (int, float)):
                            if np.isinf(value) or np.isnan(value):
                                features[processed_name] = 0.0
                                missing_count += 1
                                missing_features.append(processed_name)
                            else:
                                features[processed_name] = float(value)
                                found_count += 1
                        else:
                            features[processed_name] = 0.0
                            missing_count += 1
                            missing_features.append(processed_name)
                    except (ValueError, TypeError):
                        # If conversion fails, set to 0.0
                        features[processed_name] = 0.0
                        missing_count += 1
                        missing_features.append(processed_name)
                else:
                    # Nếu feature không có trong data, set = 0
                    features[processed_name] = 0.0
                    missing_count += 1
                    missing_features.append(processed_name)
            
            # Log feature extraction statistics (only for first few records to avoid spam)
            if self.processed_count < 5:
                logger.info(f"FEATURE EXTRACTION | Found: {found_count}/{len(self.feature_columns)} | Missing: {missing_count}/{len(self.feature_columns)}")
                if missing_count > 0 and len(missing_features) <= 10:
                    logger.info(f"Missing features: {missing_features[:10]}")
                elif missing_count > 0:
                    logger.info(f"Missing features (first 10): {missing_features[:10]}...")
                
                # Log sample of found features để verify format
                if found_count > 0:
                    sample_features = [fname for fname in self.feature_columns if fname in features and features[fname] != 0.0][:5]
                    if sample_features:
                        logger.info(f"Sample found features: {sample_features}")
                        # Check if data uses raw or processed format
                        first_feature = sample_features[0]
                        raw_name = self.processed_to_raw_mapping.get(first_feature)
                        if raw_name and raw_name in data:
                            logger.info(f"Data format: RAW (with spaces) - e.g., '{raw_name}' found")
                        elif first_feature in data:
                            logger.info(f"Data format: PROCESSED (snake_case) - e.g., '{first_feature}' found")
        else:
            # This should never happen now with hardcoded fallback, but keeping for safety
            logger.error("No feature columns available - this should not happen")
            feature_mapping = self._load_feature_mapping()
            for raw_name, processed_name in feature_mapping.items():
                if raw_name in data:
                    value = data[raw_name]
                    if isinstance(value, (int, float)):
                        if np.isinf(value) or np.isnan(value):
                            features[processed_name] = 0.0
                        else:
                            features[processed_name] = float(value)
                    else:
                        features[processed_name] = 0.0
                else:
                    features[processed_name] = 0.0

        return features

    def _encode_labels(self, label: str) -> Dict[str, Any]:
        """
        Encode labels cho CNN models
        Trả về binary classification (benign/malicious) và attack type
        """
        # Mapping labels sang binary (0: benign, 1: malicious)
        benign_labels = ['BENIGN', 'Benign', 'benign']
        is_malicious = 1 if label not in benign_labels else 0

        # Group classification
        label_lower = str(label).lower().strip()
        if label_lower in ['benign']:
            group = 'benign'
            attack_type_encoded = 0
        elif any(dos in label_lower for dos in ['dos hulk', 'dos goldeneye', 'dos slowloris', 'dos slowhttptest', 'dos attack']):
            group = 'dos'
            attack_type_encoded = 1
        elif label_lower in ['ddos', 'ddos attack-hoic', 'ddos attack-loic-udp']:
            group = 'ddos'
            attack_type_encoded = 2
        elif label_lower in ['portscan']:
            group = 'portscan'
            attack_type_encoded = 3
        else:
            group = 'other'
            attack_type_encoded = 4

        return {
            'binary_label': is_malicious,
            'attack_type_encoded': attack_type_encoded,
            'group': group,
            'original_label': label
        }

    def _convert_numpy_to_python_types(self, obj):
        """
        Recursively convert numpy types to Python native types for JSON serialization
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_to_python_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_python_types(value) for key, value in obj.items()}
        else:
            return obj

    def _preprocess_for_cnn(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess features cho CNN input
        Reshape thành (1, timesteps, features) cho 1D CNN
        Đảm bảo thứ tự features khớp với training data
        """
        # Convert to array theo đúng thứ tự feature columns từ training
        # self.feature_columns is now guaranteed to be available (either from metadata or hardcoded)
        feature_values = []
        for feature_name in self.feature_columns:
            feature_values.append(float(features.get(feature_name, 0.0)))

        X = np.array([feature_values], dtype=np.float32)

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # CRITICAL FIX: Clip extreme values BEFORE scaling
        # Problem: Scaler was trained on data with extreme values (mean up to 18M, scale up to 36M)
        # This happens because training data had Inf values converted to very large numbers
        # Solution: Clip to match training data distribution (which also had extreme values)
        # 
        # Analysis of scaler statistics:
        # - Mean range: -2264 to 18,081,024
        # - Scale range: 0.01 to 36,317,791
        # This means some features in training had values up to ~100M
        #
        # We clip to 100M to match training distribution
        CLIP_THRESHOLD = 1e8  # 100 million - matches training data range
        if np.any(np.abs(X) > CLIP_THRESHOLD):
            extreme_count = np.sum(np.abs(X) > CLIP_THRESHOLD)
            logger.warning(f"⚠️ Clipping {extreme_count} extreme values (threshold: {CLIP_THRESHOLD:.0f})")
            X = np.clip(X, -CLIP_THRESHOLD, CLIP_THRESHOLD)
        
        # Log raw features before scaling (for first 10 records to debug)
        if self.processed_count < 10:
            logger.info(f"RAW FEATURES DEBUG | Record #{self.processed_count} | Min: {X.min():.6f} | Max: {X.max():.6f} | Mean: {X.mean():.6f} | Std: {X.std():.6f}")
            # Log some sample feature values (especially high-value features)
            sample_indices = [0, 10, 20, 30, 50, 70]  # Sample features
            sample_features = [self.feature_columns[i] if i < len(self.feature_columns) else f"feature_{i}" for i in sample_indices]
            sample_values = [X[0, i] if i < X.shape[1] else 0 for i in sample_indices]
            logger.info(f"Sample raw features: {dict(zip(sample_features, [f'{v:.2f}' for v in sample_values]))}")
            
            # Check for extreme values
            if abs(X.max()) > 1000000 or abs(X.min()) > 1000000:
                logger.warning(f"⚠️ EXTREME RAW VALUES DETECTED! Max: {X.max():.2f}, Min: {X.min():.2f}")
                # Find which features have extreme values
                extreme_indices = np.where(np.abs(X[0]) > 1000000)[0]
                if len(extreme_indices) > 0:
                    extreme_features = [self.feature_columns[i] if i < len(self.feature_columns) else f"feature_{i}" for i in extreme_indices[:10]]
                    extreme_values = [X[0, i] for i in extreme_indices[:10]]
                    logger.warning(f"Extreme value features: {dict(zip(extreme_features, [f'{v:.2f}' for v in extreme_values]))}")

        # Scale features - scaler PHẢI đã được fitted
        if not self.scaler or not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
            error_msg = "❌ FATAL: Scaler not available or not fitted! Cannot scale features."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            logger.error(f"❌ Error scaling features: {e}")
            raise
        
        # Log scaled features (for first 10 records to debug)
        if self.processed_count < 10:
            logger.info(f"SCALED FEATURES DEBUG | Record #{self.processed_count} | Min: {X_scaled.min():.6f} | Max: {X_scaled.max():.6f} | Mean: {X_scaled.mean():.6f} | Std: {X_scaled.std():.6f}")
            # Verify scaler statistics
            if self.scaler and hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                logger.info(f"Scaler stats: mean range [{self.scaler.mean_.min():.2f}, {self.scaler.mean_.max():.2f}], scale range [{self.scaler.scale_.min():.2f}, {self.scaler.scale_.max():.2f}]")
            
            # Check if scaling worked correctly (StandardScaler should produce mean≈0, std≈1)
            if abs(X_scaled.mean()) > 10 or X_scaled.std() > 10:
                logger.warning(f"⚠️ SCALING MAY BE INCORRECT! Scaled mean: {X_scaled.mean():.2f}, std: {X_scaled.std():.2f} (expected: mean≈0, std≈1)")
                logger.warning(f"   Raw mean: {X.mean():.2f}, std: {X.std():.2f}")
                if self.scaler and hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                    logger.warning(f"   Scaler was fitted with mean range: [{self.scaler.mean_.min():.2f}, {self.scaler.mean_.max():.2f}]")
                    logger.warning(f"   This suggests features from packet_capture may have very different ranges than training data!")

        # Reshape cho CNN 1D: (batch_size, timesteps=1, features)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

        return X_reshaped

    def process_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Xử lý một record từ Kafka

        Args:
            record: Raw data record

        Returns:
            Processed data cho CNN hoặc None nếu lỗi
        """
        try:
            # Extract metadata
            record_id = record.get('id', f"cnn_preprocess_{self.processed_count}")
            timestamp = record.get('timestamp', pd.Timestamp.now().isoformat())

            # Extract và clean features
            raw_features = self._extract_features(record)
            if not raw_features:
                logger.warning(f"No features extracted from record {record_id}")
                return None

            # Encode labels - try multiple possible label column names
            label = None
            possible_label_keys = [' Label', 'Label', 'label', 'label_encoded', 'label_group']
            for key in possible_label_keys:
                if key in record:
                    label = record[key]
                    break
            if label is None:
                label = 'Unknown'
                logger.warning(f"Could not find label in record {record_id}, using 'Unknown'")
            label_info = self._encode_labels(label)

            # Preprocess cho CNN
            cnn_features = self._preprocess_for_cnn(raw_features)
            
            # Log feature statistics for debugging
            cnn_features_array = np.array(cnn_features)
            logger.info(f"PREPROCESS DEBUG | Record {record_id} | Label: {label_info['original_label']} | Binary: {label_info['binary_label']} | Group: {label_info['group']} | Features shape: {cnn_features_array.shape} | Min: {cnn_features_array.min():.6f} | Max: {cnn_features_array.max():.6f} | Mean: {cnn_features_array.mean():.6f} | Std: {cnn_features_array.std():.6f}")

            # Tạo processed record
            processed_record = {
                'id': record_id,
                'timestamp': timestamp,
                'original_data': record,
                'features': self._convert_numpy_to_python_types(raw_features),  # Ensure native Python types
                'cnn_features': self._convert_numpy_to_python_types(cnn_features),  # Convert to list với native Python types
                'binary_label': int(label_info['binary_label']),  # Ensure native Python int
                'attack_type_encoded': int(label_info['attack_type_encoded']),  # Ensure native Python int
                'group': label_info['group'],
                'original_label': label_info['original_label'],
                'feature_count': len(raw_features),
                'processed_by': 'cnn_preprocessing_service'
            }

            # Update attack/benign counters
            if label_info['binary_label'] == 0:
                self.benign_count += 1
            else:
                self.attack_count += 1
                # Update specific attack type counters
                if label_info['group'] == 'dos':
                    self.dos_count += 1
                elif label_info['group'] == 'ddos':
                    self.ddos_count += 1
                elif label_info['group'] == 'portscan':
                    self.portscan_count += 1

            logger.debug(f"Processed record {record_id}: {label_info['original_label']} -> binary={label_info['binary_label']}, group={label_info['group']}")
            return processed_record

        except Exception as e:
            logger.error(f"Error processing record: {e}")
            self.error_count += 1
            return None

    def send_processed_data(self, data: Dict[str, Any]):
        """
        Gửi processed data đến Kafka topic

        Args:
            data: Processed data
        """
        try:
            key = data.get('id', str(pd.Timestamp.now().timestamp()))

            future = self.producer.send(self.output_topic, value=data, key=key)
            record_metadata = future.get(timeout=10)

            logger.debug(f"Sent processed data {data['id']} to {record_metadata.topic} "
                        f"partition {record_metadata.partition} offset {record_metadata.offset}")

        except Exception as e:
            logger.error(f"Failed to send processed data: {e}")
            self.error_count += 1

    def run(self):
        """Main processing loop - chạy liên tục như data_preprocessing_service.py"""
        logger.info("Starting CNN Data Preprocessing Service")
        logger.info(f"Input topic: {self.input_topic}")
        logger.info(f"Output topic: {self.output_topic}")

        # Reset các biến thống kê khi service khởi động
        # Điều này đảm bảo không tích lũy số liệu từ các lần chạy trước
        self.processed_count = 0
        self.label_group_summary = {}
        self.label_summary = {}
        self.error_count = 0
        self.valid_records_count = 0
        self.invalid_records_count = 0
        self.start_time = datetime.now()

        # Reset attack/benign counters
        self.benign_count = 0
        self.attack_count = 0
        self.dos_count = 0
        self.ddos_count = 0
        self.portscan_count = 0

        logger.info("Reset statistics counters - starting fresh count")

        # Đọc toàn bộ backlog, không seek_to_end (giống level1_prediction_service_cnn.py)
        self.service_start_time = datetime.now()
        self.last_partition_assignment = set()
        logger.info(f"Service started at: {self.service_start_time.isoformat()}")
        logger.info("Will process all messages in backlog (no seek_to_end)")

        try:
            for message in self.consumer:
                # Kiểm tra nếu partition assignment thay đổi (rebalance) - không seek_to_end
                current_assignment = self.consumer.assignment()
                if current_assignment != self.last_partition_assignment:
                    if current_assignment:
                        logger.info(f"Partition assignment changed (rebalance detected). Continue consuming (no seek_to_end).")
                        self.service_start_time = datetime.now()
                        self.first_message_timestamp = None
                    self.last_partition_assignment = current_assignment.copy() if current_assignment else set()

                try:
                    raw_data = message.value
                    original_key = message.key

                    # Không lọc message theo timestamp - xử lý tất cả message (giống level1_prediction_service_cnn.py)
                    logger.info(f"Processing record with key: {original_key}")

                    processed_data = self.process_record(raw_data)

                    if processed_data:
                        self._update_summary(processed_data)
                        self.send_processed_data(processed_data)
                        self.valid_records_count += 1

                        attack_type = processed_data.get('group', 'unknown')
                        if attack_type in ['dos', 'ddos', 'portscan']:
                            logger.info(f"VALID ATTACK RECORD | ID: {processed_data['id']} | Type: {attack_type} | Features: {processed_data['feature_count']}")

                        if self.processed_count % 10 == 0:
                            self._log_summary()
                    else:
                        self.invalid_records_count += 1

                    try:
                        self.consumer.commit()
                    except Exception as e:
                        logger.warning(f"Failed to commit offset: {e}")

                except Exception as e:
                    logger.error(f"Error in processing loop: {e}")
                    self.error_count += 1
                    continue

        except KeyboardInterrupt:
            logger.info("Service stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.stop()

    def _update_summary(self, processed_record: Dict[str, Any]):
        """Cập nhật summary statistics"""
        self.processed_count += 1

        # Lấy group từ record
        group = processed_record.get('group', 'unknown')

        if group not in self.label_group_summary:
            self.label_group_summary[group] = 0
        self.label_group_summary[group] += 1

        # Lấy label gốc từ record (nếu có)
        label = processed_record.get('original_label', 'unknown')
        if label not in self.label_summary:
            self.label_summary[label] = 0
        self.label_summary[label] += 1

    def _log_summary(self):
        """Log summary statistics"""
        logger.info("=" * 60)
        logger.info("CNN DATA PREPROCESSING SUMMARY:")
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
        """Cleanup resources"""
        # Log final summary
        if self.processed_count > 0:
            logger.info("")
            self._log_summary()

        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()
        logger.info("CNN Data Preprocessing Service stopped")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='CNN Data Preprocessing Service')
    parser.add_argument('--kafka-servers', default='localhost:9092',
                       help='Kafka bootstrap servers')
    parser.add_argument('--input-topic', default='raw_data_event_cnn',
                       help='Input topic name')
    parser.add_argument('--output-topic', default='preprocess_data',
    parser.add_argument('--output-topic', default='preprocess_data_cnn',
                       help='Output topic name')
    parser.add_argument('--group-id', default='safenet-cnn-preprocessing-group',
                       help='Consumer group ID')

    args = parser.parse_args()

    # Tạo thư mục logs
    os.makedirs('services/logs', exist_ok=True)

    # Khởi tạo và chạy service
    service = CNNDataPreprocessingService(
        kafka_bootstrap_servers=args.kafka_servers,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        group_id=args.group_id
    )

    try:
        logger.info("Starting CNN Data Preprocessing Service...")
        service.run()
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service failed: {e}")


if __name__ == '__main__':
    main()
