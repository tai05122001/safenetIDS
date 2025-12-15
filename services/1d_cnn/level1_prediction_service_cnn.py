"""
Kafka Service cho Level 1 Prediction sử dụng 1D CNN Model.

Service này:
- Consume messages từ topic 'preprocessed_events'
- Sử dụng CNN 1D model để phân loại traffic: benign vs malicious
- Publish kết quả đến topic 'level1_predictions'

Model: CNN 1D được train từ train_level1_cnn.py
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


class Level1CNNPredictionService:
    """Service xử lý prediction Level 1 với CNN 1D."""

    def __init__(
        self,
        kafka_servers: str = "localhost:9092",
        group_id: str = "level1_cnn_service",
        input_topic: str = "cnn_preprocess_data",
        output_topic: str = "level1_predictions",
        model_path: str = "artifacts_cnn/cnn_model_best.h5",
        scaler_path: str = "artifacts_cnn/scaler.joblib",
        label_encoder_path: str = "artifacts_cnn/label_encoder.joblib",
        metadata_path: str = "artifacts_cnn/training_metadata.json",
        poll_timeout: int = 1000,
        max_retries: int = 3
    ):
        """Initialize service với các tham số cần thiết."""
        self.kafka_servers = kafka_servers
        self.group_id = group_id
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.poll_timeout = poll_timeout
        self.max_retries = max_retries

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

        # Stats
        self.processed_count = 0
        self.error_count = 0
        self.benign_count = 0
        self.malicious_count = 0
        self.attack_types_stats = {}  # Track attack types if available
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
                logging.FileHandler('services/logs/level1_prediction_cnn.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('Level1CNN')

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
                metadata = json.load(f)

            self.feature_columns = metadata.get('data_info', {}).get('feature_columns', [])
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
        """Load CNN model và preprocessing artifacts."""
        try:
            # Load feature columns first
            self.load_feature_columns()

            self.logger.info("Loading CNN model từ: %s", self.model_path)
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model = tf.keras.models.load_model(str(self.model_path))
            self.logger.info("Model loaded successfully")

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
            self.model.summary(print_fn=self.logger.info)

            return True

        except Exception as e:
            self.logger.error("Failed to load model: %s", str(e))
            return False

    def setup_kafka(self) -> bool:
        """Setup Kafka consumer và producer."""
        try:
            # Setup consumer với cấu hình giống cnn_data_preprocessing_service.py
            self.logger.info("Setting up Kafka consumer...")
            self.consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.kafka_servers,
                group_id=self.group_id,
                auto_offset_reset='earliest',  # Đọc toàn bộ backlog (đồng bộ preprocess)
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

    def _get_raw_to_processed_mapping(self) -> Dict[str, str]:
        """Get mapping từ raw CICIDS2017 names sang processed names."""
        return {
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

    def preprocess_features(self, features: Dict[str, Any]) -> Optional[np.ndarray]:
        """Preprocess features cho CNN model. Hỗ trợ cả raw format (spaces) và processed format (snake_case)."""
        try:
            # Get mapping từ raw sang processed
            raw_to_processed = self._get_raw_to_processed_mapping()

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

            # Handle NaN values (additional safety)
            if np.isnan(X).any():
                X = np.nan_to_num(X, nan=0.0)

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

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Thực hiện prediction với CNN model."""
        try:
            # Get predictions
            predictions = self.model.predict(features, verbose=0)

            if self.model.layers[-1].activation.__name__ == 'sigmoid':
                # Binary classification
                probability = float(predictions[0][0])
                predicted_class_idx = 1 if probability > 0.5 else 0
                confidence = max(probability, 1 - probability)
            else:
                # Multi-class classification
                predicted_class_idx = int(np.argmax(predictions[0]))
                confidence = float(np.max(predictions[0]))

            # Get class name
            if hasattr(self.label_encoder, 'classes_'):
                predicted_class = self.label_encoder.classes_[predicted_class_idx]
            else:
                predicted_class = f"class_{predicted_class_idx}"

            result = {
                'predicted_class': predicted_class,
                'predicted_class_idx': int(predicted_class_idx),  # Ensure native Python int
                'confidence': float(confidence),  # Ensure native Python float
                'probabilities': [float(p) for p in predictions[0]] if len(predictions[0]) <= 10 else [],  # Ensure native Python floats
                'model_type': 'cnn_1d_level1',
                'timestamp': time.time()
            }

            return result

        except Exception as e:
            self.logger.error("Error during prediction: %s", str(e))
            return {
                'error': str(e),
                'model_type': 'cnn_1d_level1',
                'timestamp': time.time()
            }

    def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Xử lý một message từ Kafka."""
        try:
            message_id = message.get('id', 'unknown')

            # Try to use preprocessed CNN features first (recommended)
            cnn_features = message.get('cnn_features')
            if cnn_features is None:
                self.logger.warning("Missing cnn_features in message %s, skipping to keep pipeline consistent", message_id)
                return None

            # Convert list back to numpy array if needed
            X = np.array(cnn_features, dtype=np.float32)
            # Ensure correct shape for CNN: (batch_size, timesteps=1, features)
            if X.ndim == 3 and X.shape[0] == 1 and X.shape[1] == 1 and X.shape[2] == 78:
                pass
            elif X.ndim == 2 and X.shape[0] == 1 and X.shape[1] == 78:
                X = X.reshape(1, 1, -1)
            elif X.ndim == 1 and X.shape[0] == 78:
                X = X.reshape(1, 1, -1)
            else:
                self.logger.warning(f"Unexpected CNN features shape: {X.shape}, expected (1, 1, 78)")
                X = X.reshape(1, 1, -1)  # Force reshape, may fail if wrong size

            self.logger.debug(f"Using preprocessed CNN features for message {message_id}, shape: {X.shape}")

            # Make prediction
            prediction_result = self.predict(X)

            # Update detection statistics
            predicted_class_idx = prediction_result.get('predicted_class_idx', 0)
            confidence = prediction_result.get('confidence', 0.0)

            # Level 1: 0 = benign, 1 = malicious
            if predicted_class_idx == 0:
                self.benign_count += 1
                detection_type = "BENIGN"
            else:
                self.malicious_count += 1
                detection_type = "MALICIOUS"

            # Log individual detections for malicious traffic
            if predicted_class_idx == 1:
                self.logger.info(f"ATTACK DETECTED | ID: {message_id} | Confidence: {confidence:.3f}")

            # Log prediction (cả benign/malicious) để đối chiếu accuracy
            self.logger.info(
                f"LEVEL1 PREDICTED | ID: {message_id} | Prediction: {detection_type.lower()} | Confidence: {confidence:.3f}"
            )

            # Combine với thông tin gốc
            result = {
                'id': message_id,
                'original_message': self._convert_numpy_to_python_types(message),  # Ensure native types
                'level1_prediction': self._convert_numpy_to_python_types(prediction_result),  # Ensure native types
                'processing_timestamp': time.time()
            }

            self.processed_count += 1

            # Log detailed stats periodically
            if self.processed_count % 1 == 0:  # Giảm từ 50 xuống 25 để log thường xuyên hơn
                elapsed = time.time() - self.start_time
                rate = self.processed_count / elapsed
                benign_rate = self.benign_count / self.processed_count * 100
                malicious_rate = self.malicious_count / self.processed_count * 100
                self.logger.info(f"LEVEL1 STATS | Total: {self.processed_count} | Benign: {self.benign_count} ({benign_rate:.1f}%) | Malicious: {self.malicious_count} ({malicious_rate:.1f}%) | Rate: {rate:.2f} msg/sec")

                # Log current summary immediately
                if self.processed_count > 0:
                    self.logger.info("")
                    self._log_summary()
                    self.logger.info("")

            return result

        except Exception as e:
            self.error_count += 1
            self.logger.error("Error processing message: %s", str(e))
            return None

    def run(self) -> None:
        """Main service loop với logic reset offset tương tự cnn_data_preprocessing_service.py"""
        self.logger.info("Starting Level 1 CNN Prediction Service")
        self.logger.info("Input topic: %s", self.input_topic)
        self.logger.info("Output topic: %s", self.output_topic)

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
        self.error_count = 0
        self.benign_count = 0
        self.malicious_count = 0
        self.attack_types_stats = {}
        self.logger.info("Reset statistics counters - starting fresh count")

        # Đọc toàn bộ backlog, không seek_to_end (đồng bộ preprocess)
        self.service_start_time = datetime.now()
        self.last_partition_assignment = set()

        self.logger.info("Service started successfully, waiting for messages...")

        try:
            while True:
                # Kiểm tra nếu partition assignment thay đổi (rebalance)
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

                            # Process message (không lọc timestamp để không bỏ lỡ dữ liệu)
                            result = self.process_message(raw_data)

                            if result is not None:
                                # Send result to output topic
                                future = self.producer.send(self.output_topic, result)
                                future.get(timeout=10)  # Wait for confirmation

                                self.logger.debug("Prediction sent to %s: %s",
                                                self.output_topic, result.get('level1_prediction', {}).get('predicted_class'))

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
        return {
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'uptime_seconds': elapsed,
            'processing_rate': self.processed_count / elapsed if elapsed > 0 else 0,
            'error_rate': self.error_count / (self.processed_count + self.error_count) if (self.processed_count + self.error_count) > 0 else 0
        }

    def _log_summary(self):
        """Log summary statistics"""
        self.logger.info("=" * 60)
        self.logger.info("LEVEL 1 PREDICTION SUMMARY (CNN):")
        self.logger.info("=" * 60)
        self.logger.info(f"Total predictions: {self.processed_count}")

        if self.processed_count > 0:
            self.logger.info("")
            self.logger.info("Prediction distribution:")
            benign_pct = self.benign_count / self.processed_count * 100
            malicious_pct = self.malicious_count / self.processed_count * 100
            self.logger.info(f"  - Benign: {self.benign_count} ({benign_pct:.1f}%)")
            self.logger.info(f"  - Attack: {self.malicious_count} ({malicious_pct:.1f}%)")

            if self.error_count > 0:
                error_pct = self.error_count / (self.processed_count + self.error_count) * 100
                self.logger.info(f"  - Errors: {self.error_count} ({error_pct:.1f}%)")

        self.logger.info("=" * 60)

    def stop(self) -> None:
        """Stop service và cleanup resources."""
        # Log final summary
        self._log_final_summary()

        self.logger.info("🛑 Stopping Level 1 CNN Prediction Service...")

        # Close consumer
        if self.consumer:
            self.consumer.close()
            self.logger.info("Consumer closed")

        # Close producer
        if self.producer:
            self.producer.close()
            self.logger.info("Producer closed")

        # Log final stats
        stats = self.get_stats()
        self.logger.info("Final stats: %s", stats)

        self.logger.info("Service stopped")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Level 1 CNN Prediction Service")
    parser.add_argument("--kafka-servers", default="localhost:9092", help="Kafka servers")
    parser.add_argument("--group-id", default="level1_cnn_service", help="Consumer group ID")
    parser.add_argument("--input-topic", default="cnn_preprocess_data", help="Input topic")
    parser.add_argument("--output-topic", default="level_1_predictions", help="Output topic")
    parser.add_argument("--model-path", default="artifacts_cnn/cnn_model_best.h5", help="Path to CNN model")
    parser.add_argument("--scaler-path", default="artifacts_cnn/scaler.joblib", help="Path to scaler")
    parser.add_argument("--label-encoder-path", default="artifacts_cnn/label_encoder.joblib", help="Path to label encoder")
    parser.add_argument("--metadata-path", default="artifacts_cnn/training_metadata.json", help="Path to training metadata")
    parser.add_argument("--poll-timeout", type=int, default=1000, help="Poll timeout in ms")

    args = parser.parse_args()

    # Create and run service
    service = Level1CNNPredictionService(
        kafka_servers=args.kafka_servers,
        group_id=args.group_id,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        label_encoder_path=args.label_encoder_path,
        metadata_path=args.metadata_path,
        poll_timeout=args.poll_timeout
    )

    service.run()


if __name__ == "__main__":
    main()
