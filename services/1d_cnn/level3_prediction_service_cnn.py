"""
Kafka Service cho Level 3 Prediction sử dụng 1D CNN Model - DoS Attack Variants Classification.

Level 3: Phân loại chi tiết loại DoS (DoS Hulk, DoS GoldenEye, DoS Slowloris, DoS Slowhttptest)
Chỉ chạy khi Level 2 = dos

Service này:
- Consume messages từ topic 'level2_predictions'
- Chỉ xử lý khi Level 2 detect "dos" attacks
- Sử dụng Advanced CNN 1D model để phân loại chi tiết DoS variants
- Publish kết quả đến topic 'level3_predictions'

Model: Advanced CNN 1D được train từ train_level3_dos_cnn.py
Mapping: label_encoder.classes_ ['2', '3', '4', '5'] -> DoS Hulk, DoS GoldenEye, DoS Slowloris, DoS Slowhttptest
"""

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Set

import joblib
import numpy as np
import tensorflow as tf
from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from kafka.errors import KafkaError


class Level3CNNPredictionService:
    """Service xử lý prediction Level 3 với Advanced CNN 1D - DoS Variants."""

    def __init__(
        self,
        kafka_servers: str = "localhost:9092",
        group_id: str = "level3_cnn_service",
        input_topic: str = "level_2_predictions",
        output_topic: str = "level_3_predictions",
        model_path: str = "artifacts_cnn_level3\\dos_classifier_cnn_final.h5",
        scaler_path: str = "artifacts_cnn_level3\\scaler.joblib",
        label_encoder_path: str = "artifacts_cnn_level3\\label_encoder.joblib",
        metadata_path: str = "artifacts_cnn_level3\\training_metadata.json",
        poll_timeout: int = 1000,
        max_retries: int = 3,
        confidence_threshold: float = 0.6,
        dos_attack_types: list = None
    ):
        """Initialize service với các tham số cần thiết."""
        self.kafka_servers = kafka_servers
        self.group_id = group_id
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.poll_timeout = poll_timeout
        self.max_retries = max_retries
        self.confidence_threshold = confidence_threshold

        # DoS attack types to monitor
        self.dos_attack_types = dos_attack_types or [
            'dos', 'ddos', 'hulk', 'goldeneye', 'slowloris', 'slowhttptest'
        ]

        # DoS variant mapping khớp với Level 3 CNN training labels
        # label_encoder.classes_ = ["2", "3", "4", "5"] (DoS Hulk, DoS GoldenEye, DoS Slowloris, DoS Slowhttptest)
        self.dos_variant_labels = {
            # Integer mappings (class indices)
            0: "DoS Hulk",           # label_encoder.classes_[0] = "2" -> DoS Hulk
            1: "DoS GoldenEye",      # label_encoder.classes_[1] = "3" -> DoS GoldenEye
            2: "DoS Slowloris",      # label_encoder.classes_[2] = "4" -> DoS Slowloris
            3: "DoS Slowhttptest",   # label_encoder.classes_[3] = "5" -> DoS Slowhttptest
            # String mappings (from label_encoder.classes_)
            "2": "DoS Hulk",          # label_encoder.classes_[0]
            "3": "DoS GoldenEye",     # label_encoder.classes_[1]
            "4": "DoS Slowloris",     # label_encoder.classes_[2]
            "5": "DoS Slowhttptest",  # label_encoder.classes_[3]
            # Legacy mappings for backward compatibility
            "DoS Hulk": "DoS Hulk",
            "DoS GoldenEye": "DoS GoldenEye",
            "DoS Slowloris": "DoS Slowloris",
            "DoS Slowhttptest": "DoS Slowhttptest"
        }

        # Model artifacts
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.label_encoder_path = Path(label_encoder_path)
        self.metadata_path = Path(metadata_path)

        # Components
        self.consumer: Optional[KafkaConsumer] = None
        self.producer: Optional[KafkaProducer] = None
        self.model: Optional[tf.keras.Model] = None
        self.scaler = None
        self.label_encoder = None

        # Feature columns for consistent ordering
        self.feature_columns = None
        self.metadata = None

        # Stats
        self.processed_count = 0
        self.skipped_count = 0
        self.error_count = 0
        self.dos_detected_count = 0
        self.variant_classified_count = 0
        self.dos_variants_stats = {}
        self.confidence_summary = {}
        self.start_time = time.time()

        # Timestamp khi service khởi động để chỉ xử lý message mới
        self.service_start_time = None
        # Lưu partition assignment để detect rebalance
        self.last_partition_assignment: Set[TopicPartition] = set()
        # Lưu timestamp của message đầu tiên được xử lý
        # Điều này đảm bảo chỉ xử lý message có timestamp >= message đầu tiên
        self.first_message_timestamp = None

        # Setup logging
        self._setup_logging()

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('services/logs/level3_prediction_cnn.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('Level3CNN')

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()

    def load_feature_columns(self) -> bool:
        """Load feature columns từ training metadata để đảm bảo thứ tự đúng."""
        try:
            self.logger.info("Loading feature columns từ: %s", self.metadata_path)
            if not self.metadata_path.exists():
                self.logger.warning("Training metadata not found, using default feature order")
                return False

            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

            self.feature_columns = self.metadata.get('data_info', {}).get('feature_columns', [])
            if self.feature_columns:
                self.logger.info(f"Loaded {len(self.feature_columns)} feature columns from training metadata")
                self.logger.info(f"First 5 features: {self.feature_columns[:5]}")
                return True
            else:
                self.logger.warning("No feature columns found in metadata")
                return False

        except Exception as e:
            self.logger.error("Failed to load feature columns: %s", str(e))
            return False

    def load_model(self) -> bool:
        """Load Advanced CNN model và preprocessing artifacts."""
        try:
            # Load feature columns first
            self.load_feature_columns()

            self.logger.info("Loading Level 3 DoS CNN model từ: %s", self.model_path)
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
            self.logger.info("Level 3 DoS Model loaded successfully")

            # Load scaler
            self.logger.info("Loading scaler từ: %s", self.scaler_path)
            if not self.scaler_path.exists():
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

            self.scaler = joblib.load(self.scaler_path)
            self.logger.info("Scaler loaded successfully")

            # Load label encoder
            self.logger.info("Loading label encoder từ: %s", self.label_encoder_path)
            if not self.label_encoder_path.exists():
                raise FileNotFoundError(f"Label encoder file not found: {self.label_encoder_path}")

            self.label_encoder = joblib.load(self.label_encoder_path)
            self.logger.info("Label encoder loaded successfully")

            # Log model info
            self.logger.info("DoS Variants: %s", list(self.label_encoder.classes_))
            self.model.summary(print_fn=self.logger.info)

            # Initialize DoS variants stats with string keys
            for variant in self.label_encoder.classes_:
                self.dos_variants_stats[str(variant)] = 0
                self.confidence_summary[str(variant)] = []

            return True

        except Exception as e:
            self.logger.error("Failed to load Level 3 DoS model: %s", str(e))
            return False

    def setup_kafka(self) -> bool:
        """Setup Kafka consumer và producer."""
        try:
            # Setup consumer với cấu hình giống các service khác
            self.logger.info("Setting up Kafka consumer...")
            self.consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.kafka_servers,
                group_id=self.group_id,
                auto_offset_reset='earliest',  # Đọc toàn bộ backlog
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                session_timeout_ms=30000,
                heartbeat_interval_ms=3000,
                max_poll_records=100,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                consumer_timeout_ms=self.poll_timeout
            )
            self.logger.info("Consumer setup completed")

            # Setup producer
            self.logger.info("Setting up Kafka producer...")
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                acks='all',
                retries=3,
                linger_ms=5,
                batch_size=32768,
                buffer_memory=67108864
            )
            self.logger.info("Producer setup completed")

            return True

        except Exception as e:
            self.logger.error("Failed to setup Kafka: %s", str(e))
            return False

    def is_dos_attack(self, level2_prediction: Dict[str, Any]) -> bool:
        """Kiểm tra xem có phải là DoS attack không từ Level 2 prediction."""
        try:
            predicted_attack_type = level2_prediction.get('predicted_attack_type', '')
            confidence = level2_prediction.get('confidence', 0.0)

            # Level 2 output: "dos", "ddos", "portscan" hoặc có thể là số
            # Chỉ xử lý khi Level 2 detect "dos" hoặc class_idx = 0
            if isinstance(predicted_attack_type, str):
                is_dos = predicted_attack_type.lower() == "dos"
            elif isinstance(predicted_attack_type, (int, float)):
                # Có thể là class_idx từ label_encoder
                is_dos = int(predicted_attack_type) == 0  # 0 = dos
            else:
                is_dos = str(predicted_attack_type).lower() == "dos"

            has_confidence = confidence >= self.confidence_threshold

            should_process = is_dos and has_confidence

            if not should_process:
                reason = []
                if not is_dos:
                    reason.append(f"not DoS attack (Level 2 type: {predicted_attack_type})")
                if not has_confidence:
                    reason.append(f"low confidence ({confidence:.3f})")
                self.logger.debug("Skipping message: %s", ", ".join(reason))

            return should_process

        except Exception as e:
            self.logger.error("Error checking is_dos_attack: %s", str(e))
            return False

    def preprocess_features(self, features: Dict[str, Any]) -> Optional[np.ndarray]:
        """Preprocess features cho Level 3 DoS CNN model với đúng thứ tự từ training."""
        try:
            # Get mapping từ raw CICIDS2017 names sang processed names (tương tự Level 1)
            raw_to_processed = {
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

            # Convert features dict to array theo đúng thứ tự training
            if self.feature_columns:
                # Use feature columns from metadata for consistent ordering
                feature_values = []
                for feature_name in self.feature_columns:
                    value = None

                    # Try different possible column name formats
                    possible_names = [
                        feature_name,  # snake_case (processed)
                        raw_to_processed.get(feature_name, feature_name)  # raw with spaces (fallback to same name if not found)
                    ]

                    # Remove duplicates
                    possible_names = list(set(possible_names))

                    # Try to find the value in data
                    for name in possible_names:
                        if name in features:
                            value = features[name]
                            break

                    # Convert to float
                    if value is not None:
                        try:
                            if isinstance(value, str):
                                if value.strip() == '':
                                    feature_values.append(0.0)
                                else:
                                    numeric_value = float(value)
                                    if np.isinf(numeric_value) or np.isnan(numeric_value):
                                        feature_values.append(0.0)
                                    else:
                                        feature_values.append(numeric_value)
                            elif isinstance(value, (int, float)):
                                if np.isinf(value) or np.isnan(value):
                                    feature_values.append(0.0)
                                else:
                                    feature_values.append(float(value))
                            else:
                                feature_values.append(0.0)
                        except (ValueError, TypeError):
                            feature_values.append(0.0)
                    else:
                        feature_values.append(0.0)
            else:
                # Fallback: sort alphabetically (not recommended)
                self.logger.warning("No feature columns loaded, using alphabetical sorting (may cause prediction errors)")
                feature_values = []
                for key, value in sorted(features.items()):
                    if isinstance(value, (int, float)):
                        if np.isinf(value) or np.isnan(value):
                            feature_values.append(0.0)
                        else:
                            feature_values.append(float(value))
                    else:
                        feature_values.append(0.0)

            X = np.array([feature_values], dtype=np.float32)

            # Advanced NaN handling (median imputation for DoS features)
            if np.isnan(X).any():
                for col_idx in range(X.shape[1]):
                    col_median = np.nanmedian(X[:, col_idx])
                    X[np.isnan(X[:, col_idx]), col_idx] = col_median

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Reshape cho CNN 1D: (batch_size, timesteps=1, features)
            X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

            return X_reshaped

        except Exception as e:
            self.logger.error("Error preprocessing features: %s", str(e))
            return None

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

    def predict_dos_variant(self, features: np.ndarray) -> Dict[str, Any]:
        """Thực hiện prediction DoS variant với Advanced CNN model."""
        try:
            # Get predictions
            predictions = self.model.predict(features, verbose=0)

            # Multi-class classification for DoS variants
            predicted_class_idx = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))

            # Get DoS variant name (readable label instead of class index)
            if hasattr(self.label_encoder, 'classes_'):
                class_label = str(self.label_encoder.classes_[predicted_class_idx])
                predicted_dos_variant = self.dos_variant_labels.get(class_label, self.dos_variant_labels.get(predicted_class_idx, class_label))
            else:
                predicted_dos_variant = self.dos_variant_labels.get(predicted_class_idx, f"dos_variant_{predicted_class_idx}")

            # Get top-k predictions for DoS variants
            top_k = 3
            top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]
            top_k_predictions = []

            for idx in top_k_indices:
                if hasattr(self.label_encoder, 'classes_'):
                    class_label = str(self.label_encoder.classes_[idx])
                    dos_variant = self.dos_variant_labels.get(class_label, self.dos_variant_labels.get(idx, class_label))
                else:
                    dos_variant = self.dos_variant_labels.get(idx, f"dos_variant_{idx}")
                prob = float(predictions[0][idx])
                top_k_predictions.append({
                    'dos_variant': dos_variant,
                    'probability': prob
                })

            # Update stats using class label as key (not readable name)
            self.dos_variants_stats[class_label] += 1

            result = {
                'predicted_dos_variant': predicted_dos_variant,
                'predicted_class_idx': predicted_class_idx,
                'confidence': confidence,
                'top_k_predictions': top_k_predictions,
                'all_probabilities': predictions[0].tolist(),
                'model_type': 'cnn_1d_level3_dos_variants',
                'severity_assessment': self._assess_severity(predicted_dos_variant, confidence),
                'timestamp': time.time()
            }

            return self._convert_numpy_to_python_types(result)

        except Exception as e:
            self.logger.error("Error during DoS variant prediction: %s", str(e))
            return self._convert_numpy_to_python_types({
                'error': str(e),
                'model_type': 'cnn_1d_level3_dos_variants',
                'timestamp': time.time()
            })

    def _assess_severity(self, dos_variant: str, confidence: float) -> Dict[str, Any]:
        """Đánh giá mức độ nghiêm trọng của DoS attack."""
        try:
            # Define severity scores cho different DoS variants
            # Mapping khớp với label_encoder.classes_ từ training: ['2', '3', '4', '5']
            # Và các readable names từ dos_variant_labels mapping
            severity_map = {
                # Class indices từ label_encoder.classes_
                '2': {'base_severity': 'high', 'impact_score': 9, 'description': 'DoS Hulk - HTTP Flood - High bandwidth consumption'},
                '3': {'base_severity': 'high', 'impact_score': 8, 'description': 'DoS GoldenEye - Slowloris variant - Resource exhaustion'},
                '4': {'base_severity': 'medium', 'impact_score': 7, 'description': 'DoS Slowloris - Slow POST - Connection pool exhaustion'},
                '5': {'base_severity': 'medium', 'impact_score': 6, 'description': 'DoS Slowhttptest - Slow HTTP headers - Memory exhaustion'},
                # Readable names từ dos_variant_labels mapping
                'dos hulk': {'base_severity': 'high', 'impact_score': 9, 'description': 'DoS Hulk - HTTP Flood - High bandwidth consumption'},
                'dos goldeneye': {'base_severity': 'high', 'impact_score': 8, 'description': 'DoS GoldenEye - Slowloris variant - Resource exhaustion'},
                'dos slowloris': {'base_severity': 'medium', 'impact_score': 7, 'description': 'DoS Slowloris - Slow POST - Connection pool exhaustion'},
                'dos slowhttptest': {'base_severity': 'medium', 'impact_score': 6, 'description': 'DoS Slowhttptest - Slow HTTP headers - Memory exhaustion'},
                # Legacy mappings for backward compatibility
                'hulk': {'base_severity': 'high', 'impact_score': 9, 'description': 'HTTP Flood - High bandwidth consumption'},
                'goldeneye': {'base_severity': 'high', 'impact_score': 8, 'description': 'Slowloris variant - Resource exhaustion'},
                'slowloris': {'base_severity': 'medium', 'impact_score': 7, 'description': 'Slow POST - Connection pool exhaustion'},
                'slowhttptest': {'base_severity': 'medium', 'impact_score': 6, 'description': 'Slow HTTP headers - Memory exhaustion'},
                'dos': {'base_severity': 'high', 'impact_score': 8, 'description': 'Generic DoS - Variable impact'},
                'ddos': {'base_severity': 'critical', 'impact_score': 10, 'description': 'Distributed DoS - Maximum impact'}
            }

            variant_info = severity_map.get(dos_variant.lower(), {
                'base_severity': 'medium',
                'impact_score': 5,
                'description': 'Unknown DoS variant'
            })

            # Adjust severity based on confidence
            adjusted_severity = variant_info['base_severity']
            if confidence > 0.9:
                # High confidence - maintain base severity
                pass
            elif confidence > 0.7:
                # Medium confidence - slightly reduce severity
                if adjusted_severity == 'critical':
                    adjusted_severity = 'high'
                elif adjusted_severity == 'high':
                    adjusted_severity = 'medium'
            else:
                # Low confidence - reduce severity significantly
                adjusted_severity = 'low'

            return {
                'severity_level': adjusted_severity,
                'impact_score': variant_info['impact_score'],
                'confidence_adjusted': confidence > 0.7,
                'description': variant_info['description'],
                'recommended_actions': self._get_recommended_actions(adjusted_severity, dos_variant)
            }

        except Exception as e:
            self.logger.error("Error assessing severity: %s", str(e))
            return {
                'severity_level': 'unknown',
                'impact_score': 0,
                'confidence_adjusted': False,
                'description': 'Severity assessment failed',
                'recommended_actions': ['Investigate manually']
            }

    def _get_recommended_actions(self, severity: str, dos_variant: str) -> list:
        """Tạo danh sách recommended actions dựa trên severity và attack type."""
        base_actions = [
            "Monitor network traffic patterns",
            "Check server resource utilization",
            "Review firewall rules"
        ]

        if severity == 'critical':
            base_actions.extend([
                "IMMEDIATE ACTION REQUIRED",
                "Isolate affected systems",
                "Contact security incident response team",
                "Consider service shutdown if necessary",
                "Notify network operations center"
            ])
        elif severity == 'high':
            base_actions.extend([
                "Increase monitoring frequency",
                "Apply rate limiting rules",
                "Check for botnet indicators",
                "Prepare incident response plan"
            ])
        elif severity == 'medium':
            base_actions.extend([
                "Log detailed traffic analysis",
                "Review recent security updates",
                "Monitor for attack progression"
            ])
        else:  # low
            base_actions.extend([
                "Continue normal monitoring",
                "Add to threat intelligence feeds",
                "Review false positive indicators"
            ])

        # Specific actions based on DoS variant
        dos_variant_lower = dos_variant.lower()
        if 'slow' in dos_variant_lower or 'slowhttptest' in dos_variant_lower:
            base_actions.append("Check connection pool limits")
        elif 'hulk' in dos_variant_lower:
            base_actions.append("Implement HTTP request throttling")
        elif 'goldeneye' in dos_variant_lower:
            base_actions.append("Review slowloris protection mechanisms")

        return base_actions

    def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Xử lý một message từ Kafka."""
        try:
            message_id = message.get('id', 'unknown')
            level2_prediction = message.get('level2_prediction', {})

            self.logger.debug("Processing message ID: %s", message_id)

            # Check if we should process this message (only DoS attacks)
            if not self.is_dos_attack(level2_prediction):
                self.skipped_count += 1
                return None  # Skip non-DoS attacks

            # Always preprocess from raw features with Level 3 metadata
            # DO NOT use cnn_features from upstream as it uses Level 1 feature ordering
            # Navigate through nested structure: level2_message -> level1_message -> original_message -> features
            original_message = message.get('original_message', {})
            if isinstance(original_message, dict):
                level1_message = original_message.get('original_message', {})
                if isinstance(level1_message, dict):
                    original_features = level1_message.get('features', {})

                    if not original_features:
                        self.logger.warning("No raw features found for message %s", message_id)
                        self.error_count += 1
                        return None

                    # Preprocess with Level 3 feature ordering (required for Level 3 model)
                    X = self.preprocess_features(original_features)
                    if X is None:
                        self.logger.warning("Failed to preprocess features for message %s", message_id)
                        self.error_count += 1
                        return None

                    self.logger.debug("Preprocessed features with Level 3 metadata for message %s", message_id)
                else:
                    self.logger.warning("Invalid level1 message structure for message %s", message_id)
                    self.error_count += 1
                    return None
            else:
                self.logger.warning("Invalid message structure for message %s", message_id)
                self.error_count += 1
                return None

            # Make DoS variant prediction
            prediction_result = self.predict_dos_variant(X)

            # Combine với thông tin gốc
            result = {
                'id': message_id,
                'original_message': self._convert_numpy_to_python_types(message),  # Ensure native types
                'level3_prediction': self._convert_numpy_to_python_types(prediction_result),  # Ensure native types
                'processing_timestamp': time.time()
            }

            self.processed_count += 1
            self.dos_detected_count += 1
            self.variant_classified_count += 1

            # Log DoS variant detection with severity
            dos_variant = prediction_result.get('predicted_dos_variant', 'unknown')
            confidence = prediction_result.get('confidence', 0.0)
            severity = prediction_result.get('severity_assessment', {}).get('severity_level', 'unknown')
            predicted_class_idx = prediction_result.get('predicted_class_idx', 0)

            # Get class label for stats (use predicted_class_idx to get class label)
            class_label_for_stats = str(self.label_encoder.classes_[predicted_class_idx]) if hasattr(self.label_encoder, 'classes_') else str(predicted_class_idx)

            if confidence > 0:
                self.confidence_summary.setdefault(class_label_for_stats, []).append(confidence)

            severity_text = f"[{severity.upper()}]"
            self.logger.info(f"DoS Variant: {dos_variant} | Severity: {severity} | Confidence: {confidence:.3f}")

            # Log stats periodically
            if self.variant_classified_count % 1 == 0:
                self._log_summary()

            return result

        except Exception as e:
            self.error_count += 1
            self.logger.error("Error processing message: %s", str(e))
            return None

    def run(self) -> None:
        """Main service loop với logic reset offset tương tự các service khác."""
        self.logger.info("🚀 Starting Level 3 CNN DoS Variants Prediction Service")
        self.logger.info("Input topic: %s", self.input_topic)
        self.logger.info("Output topic: %s", self.output_topic)
        self.logger.info("Confidence threshold: %.2f", self.confidence_threshold)
        self.logger.info("Processing only 'dos' attacks from Level 2")
        self.logger.info("DoS variants: DoS Hulk(2), DoS GoldenEye(3), DoS Slowloris(4), DoS Slowhttptest(5)")

        # Load model
        if not self.load_model():
            self.logger.error("Cannot start service without model")
            return

        # Setup Kafka
        if not self.setup_kafka():
            self.logger.error("Cannot start service without Kafka connection")
            return

        # Reset statistics counters khi service khởi động
        # Điều này đảm bảo không tích lũy số liệu từ các lần chạy trước
        self.processed_count = 0
        self.skipped_count = 0
        self.error_count = 0
        self.dos_detected_count = 0
        self.variant_classified_count = 0
        # Reset dos_variants_stats with string keys
        self.dos_variants_stats = {str(variant): 0 for variant in self.label_encoder.classes_}
        self.confidence_summary = {str(variant): [] for variant in self.label_encoder.classes_}
        self.logger.info("Reset statistics counters - starting fresh count")

        # Đọc toàn bộ backlog, không seek_to_end (đồng bộ RF logic)
        self.service_start_time = datetime.now()
        self.last_partition_assignment = set()

        self.logger.info("Service started successfully, waiting for messages...")

        try:
            while True:
                # Kiểm tra nếu partition assignment thay đổi (rebalance) và tiếp tục (không seek_to_end)
                current_assignment = self.consumer.assignment()
                if current_assignment != self.last_partition_assignment:
                    if current_assignment:
                        self.logger.info(f"Partition assignment changed (rebalance detected). Continue consuming (no seek_to_end).")
                        self.service_start_time = datetime.now()
                        self.first_message_timestamp = None
                    self.last_partition_assignment = current_assignment.copy() if current_assignment else set()

                try:
                    # Poll for messages
                    message_batch = self.consumer.poll(timeout_ms=self.poll_timeout)

                    if not message_batch:
                        continue

                    # Process each message
                    for topic_partition, messages in message_batch.items():
                        for message in messages:
                            # Lấy dữ liệu từ message
                            raw_data = message.value

                            # Process message (không lọc timestamp để không bỏ lỡ backlog)
                            result = self.process_message(raw_data)

                            if result is not None:
                                # Send result to output topic
                                future = self.producer.send(self.output_topic, result)
                                future.get(timeout=10)  # Wait for confirmation

                                dos_variant = result.get('level3_prediction', {}).get('predicted_dos_variant', 'unknown')
                                severity = result.get('level3_prediction', {}).get('severity_assessment', {}).get('severity_level', 'unknown')
                                self.logger.debug("DoS variant prediction sent: %s (Severity: %s)", dos_variant, severity)

                            # Commit offset thủ công sau mỗi message để đảm bảo không đọc lại
                            try:
                                self.consumer.commit()
                            except Exception as e:
                                self.logger.warning(f"Failed to commit offset: {e}")

                except Exception as e:
                    self.logger.error("Error in message processing loop: %s", str(e))
                    time.sleep(1)  # Brief pause before retry

        except KeyboardInterrupt:
            self.logger.info("Service interrupted by user")
        except Exception as e:
            self.logger.error("Unexpected error: %s", str(e))
        finally:
            self.stop()

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        elapsed = time.time() - self.start_time
        total_messages = self.processed_count + self.skipped_count + self.error_count
        return {
            'processed_dos_attacks': self.processed_count,
            'skipped_non_dos': self.skipped_count,
            'error_count': self.error_count,
            'dos_detected_count': self.dos_detected_count,
            'variant_classified_count': self.variant_classified_count,
            'total_messages': total_messages,
            'dos_variants_breakdown': self.dos_variants_stats,
            'uptime_seconds': elapsed,
            'processing_rate': self.processed_count / elapsed if elapsed > 0 else 0,
            'error_rate': self.error_count / total_messages if total_messages > 0 else 0,
            'dos_detection_rate': self.dos_detected_count / total_messages if total_messages > 0 else 0
        }

    def _log_final_summary(self):
        """Log summary statistics"""
        self.logger.info("=" * 60)
        self.logger.info("LEVEL 3 PREDICTION SUMMARY (CNN - DoS Detail):")
        self.logger.info("=" * 60)
        self.logger.info(f"Total predictions: {self.processed_count}")

        if self.processed_count > 0:
            self.logger.info("")
            self.logger.info("DoS subtype distribution:")
            for label, count in sorted(self.dos_variants_stats.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    percentage = (count / self.processed_count * 100) if self.processed_count > 0 else 0
                    conf_list = self.confidence_summary.get(label, [])
                    avg_conf = sum(conf_list) / len(conf_list) if conf_list else 0.0
                    self.logger.info(f"  - {label}: {count} ({percentage:.1f}%) - Avg confidence: {avg_conf:.3f}")

        self.logger.info("=" * 60)

    def _log_summary(self):
        """Log periodic summary (RF-style) with counts, percentages, avg confidence."""
        self.logger.info("=" * 60)
        self.logger.info("LEVEL 3 PREDICTION SUMMARY (CNN - DoS Detail):")
        self.logger.info("=" * 60)
        self.logger.info(f"Total predictions: {self.processed_count}")
        self.logger.info("")
        self.logger.info("DoS subtype distribution:")
        for label, count in sorted(self.dos_variants_stats.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                percentage = (count / self.processed_count * 100) if self.processed_count > 0 else 0
                conf_list = self.confidence_summary.get(label, [])
                avg_conf = sum(conf_list) / len(conf_list) if conf_list else 0.0
                self.logger.info(f"  - {label}: {count} ({percentage:.1f}%) - Avg confidence: {avg_conf:.3f}")
        self.logger.info("=" * 60)

    def stop(self) -> None:
        """Stop service và cleanup resources."""
        # Log final summary
        self._log_final_summary()

        self.logger.info("🛑 Stopping Level 3 CNN DoS Variants Prediction Service...")

        # Close consumer
        if self.consumer:
            self.consumer.close()
            self.logger.info("Consumer closed")

        # Close producer
        if self.producer:
            self.producer.close()
            self.logger.info("Producer closed")

        self.logger.info("Service stopped")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Level 3 CNN DoS Variants Prediction Service")
    parser.add_argument("--kafka-servers", default="localhost:9092", help="Kafka servers")
    parser.add_argument("--group-id", default="level3_cnn_service", help="Consumer group ID")
    parser.add_argument("--input-topic", default="level_2_predictions", help="Input topic")
    parser.add_argument("--output-topic", default="level_3_predictions", help="Output topic")
    parser.add_argument("--model-path", default="artifacts_cnn_level3/dos_classifier_cnn_best.h5", help="Path to CNN model")
    parser.add_argument("--scaler-path", default="artifacts_cnn_level3/scaler.joblib", help="Path to scaler")
    parser.add_argument("--label-encoder-path", default="artifacts_cnn_level3/label_encoder.joblib", help="Path to label encoder")
    parser.add_argument("--metadata-path", default="artifacts_cnn_level3/training_metadata.json", help="Path to training metadata")
    parser.add_argument("--confidence-threshold", type=float, default=0.6, help="Confidence threshold for processing")
    parser.add_argument("--poll-timeout", type=int, default=1000, help="Poll timeout in ms")
    parser.add_argument("--dos-types", nargs="+", default=['dos', 'ddos', 'hulk', 'goldeneye', 'slowloris', 'slowhttptest'],
                       help="DoS attack types to monitor")

    args = parser.parse_args()

    # Create and run service
    service = Level3CNNPredictionService(
        kafka_servers=args.kafka_servers,
        group_id=args.group_id,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        label_encoder_path=args.label_encoder_path,
        metadata_path=args.metadata_path,
        confidence_threshold=args.confidence_threshold,
        poll_timeout=args.poll_timeout,
        dos_attack_types=args.dos_types
    )

    service.run()


if __name__ == "__main__":
    main()
