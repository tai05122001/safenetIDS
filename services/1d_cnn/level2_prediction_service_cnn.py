"""
Kafka Service cho Level 2 Prediction sử dụng 1D CNN Model - Attack Types Classification.

Service này:
- Consume messages từ topic 'level1_predictions'
- Chỉ xử lý khi Level 1 detect malicious traffic
- Sử dụng CNN 1D model để phân loại attack types
- Publish kết quả đến topic 'level2_predictions'

Model: CNN 1D được train từ train_level2_attack_types_cnn.py
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


class Level2CNNPredictionService:
    """Service xử lý prediction Level 2 với CNN 1D - Attack Types Classification.

    KHỚP VỚI TRAINING LABELS: Chỉ phân loại malicious traffic (Level 1 đã filter benign).

    Mapping chính xác với Level 2 CNN training (filter_malicious_only=True):
    label_encoder.classes_ = ["0", "1", "2"] (label_attack_type_encoded)
    - Class 0 (label "0"): "dos" - DoS attacks (252661 samples)
    - Class 1 (label "1"): "ddos" - DDoS attacks (128027 samples)
    - Class 2 (label "2"): "portscan" - Port scanning attacks (158930 samples)

    WARNING: Mapping này PHẢI khớp với label_encoder.classes_ từ training script!
    """

    def __init__(
        self,
        kafka_servers: str = "localhost:9092",
        group_id: str = "level2_cnn_service",
        input_topic: str = "level_1_predictions",
        output_topic: str = "level_2_predictions",
        model_path: str = "artifacts_cnn_level2/attack_classifier_cnn_best.h5",
        scaler_path: str = "artifacts_cnn_level2/scaler.joblib",
        label_encoder_path: str = "artifacts_cnn_level2/label_encoder.joblib",
        poll_timeout: int = 1000,
        max_retries: int = 3,
        confidence_threshold: float = 0.5
    ):
        """Initialize service với các tham số cần thiết."""
        self.kafka_servers = kafka_servers
        self.group_id = group_id
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.poll_timeout = poll_timeout
        self.max_retries = max_retries
        self.confidence_threshold = confidence_threshold

        # Model artifacts
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.label_encoder_path = Path(label_encoder_path)

        # Components
        self.consumer: Optional[KafkaConsumer] = None
        self.producer: Optional[KafkaProducer] = None
        self.model: Optional[tf.keras.Model] = None
        self.scaler = None
        self.label_encoder = None

        # Stats
        self.processed_count = 0
        self.skipped_count = 0
        self.error_count = 0
        self.attack_detected_count = 0
        self.attack_types_stats = {}  # Track attack types detected
        self.attack_confidence_summary = {}  # Track confidence per attack type
        self.start_time = time.time()

        # Attack type mapping khớp với Level 2 CNN training labels
        # Mapping thực tế từ label_encoder.classes_ = ["0", "1", "2"] (label_attack_type_encoded)
        # Dựa trên dataset CICIDS2017 analysis:
        # label_attack_type_encoded 0 -> dos attacks (252661 samples)
        # label_attack_type_encoded 1 -> ddos attacks (128027 samples)
        # label_attack_type_encoded 2 -> portscan attacks (158930 samples)
        self.attack_type_labels = {
            # Integer mappings (class indices) - khớp với label_encoder.classes_
            0: "dos",             # label_encoder.classes_[0] = "0" -> DoS attacks
            1: "ddos",            # label_encoder.classes_[1] = "1" -> DDoS attacks
            2: "portscan",        # label_encoder.classes_[2] = "2" -> Port scanning attacks
            # String mappings cho trường hợp label_encoder trả về tên thay vì số
            "0": "dos",            # label_encoder.classes_[0]
            "1": "ddos",           # label_encoder.classes_[1]
            "2": "portscan",       # label_encoder.classes_[2]
            # Các attack types cụ thể từ CICIDS2017 dataset (fallback mapping)
            "DoS Hulk": "DoS Hulk",
            "DoS GoldenEye": "DoS GoldenEye",
            "DoS Slowloris": "DoS Slowloris",
            "DoS Slowhttptest": "DoS Slowhttptest",
            "DDoS": "DDoS",
            "DDoS attack-HOIC": "DDoS HOIC",
            "DDoS attack-LOIC-UDP": "DDoS LOIC-UDP",
            "PortScan": "PortScan",  # SCANNING ATTACK: Network port scanning
            # Legacy mappings for backward compatibility
            "DDoS": "ddos",
            "PortScan": "portscan"
        }

        # Timestamp khi service khởi động để chỉ xử lý message mới
        self.service_start_time = None
        # Lưu partition assignment để detect rebalance
        self.last_partition_assignment: Set[TopicPartition] = set()
        # Lưu timestamp của message đầu tiên được xử lý
        # Điều này đảm bảo chỉ xử lý message có timestamp >= message đầu tiên
        self.first_message_timestamp = None

        # Setup logging
        self._setup_logging()

        # Log important mapping info
        self.logger.info("Level 2 CNN Attack Type Mapping (khớp với training labels):")
        self.logger.info("  label_encoder.classes_ = ['0', '1', '2'] (label_attack_type_encoded)")
        self.logger.info("  Class 0 (label '0') → dos (252661 samples)")
        self.logger.info("  Class 1 (label '1') → ddos (128027 samples)")
        self.logger.info("  Class 2 (label '2') → portscan (158930 samples)")
        self.logger.info("WARNING: Mapping này PHẢI khớp với label_encoder từ training!")

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('services/logs/level2_prediction_cnn.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('Level2CNN')

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()

    def load_model(self) -> bool:
        """Load CNN model và preprocessing artifacts."""
        try:
            self.logger.info("Loading Level 2 CNN model từ: %s", self.model_path)
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model = tf.keras.models.load_model(str(self.model_path))
            self.logger.info("Level 2 Model loaded successfully")

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
            self.logger.info("Attack Types: %s", list(self.label_encoder.classes_))
            self.model.summary(print_fn=self.logger.info)

            return True

        except Exception as e:
            self.logger.error("Failed to load Level 2 model: %s", str(e))
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
                auto_offset_reset='earliest',  # Đọc toàn bộ backlog (đồng bộ RF logic)
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

    def should_process_message(self, level1_prediction: Dict[str, Any]) -> bool:
        """Kiểm tra xem có nên xử lý message này không dựa trên Level 1 prediction."""
        try:
            predicted_class = level1_prediction.get('predicted_class', '')
            predicted_class_idx = level1_prediction.get('predicted_class_idx', 0)
            confidence = level1_prediction.get('confidence', 0.0)

            # Handle both string class names and numeric class indices
            # Level 1 CNN typically returns: 0 = benign, 1 = malicious
            if isinstance(predicted_class, (int, float)):
                # predicted_class is numeric (class index)
                is_malicious = int(predicted_class) != 0  # 0 = benign, others = malicious
            elif isinstance(predicted_class, str):
                # predicted_class is string (class name)
                is_malicious = predicted_class.lower() not in ['benign', 'normal', 'class_0']
            else:
                # Fallback to class index
                is_malicious = int(predicted_class_idx) != 0

            has_confidence = confidence >= self.confidence_threshold

            should_process = is_malicious and has_confidence

            if not should_process:
                reason = []
                if not is_malicious:
                    reason.append(f"benign traffic (class: {predicted_class}, idx: {predicted_class_idx})")
                if not has_confidence:
                    reason.append(f"low confidence ({confidence:.3f})")
                self.logger.debug("Skipping message: %s", ", ".join(reason))

            return should_process

        except Exception as e:
            self.logger.error("Error checking should_process: %s", str(e))
            return False

    def preprocess_features(self, features: Dict[str, Any]) -> Optional[np.ndarray]:
        """Preprocess features cho Level 2 CNN model."""
        try:
            # Convert features dict to array
            feature_values = []
            for key, value in sorted(features.items()):  # Sort để đảm bảo thứ tự
                if isinstance(value, (int, float)):
                    feature_values.append(float(value))
                else:
                    # Handle non-numeric values
                    feature_values.append(0.0)

            X = np.array([feature_values], dtype=np.float32)

            # Handle NaN values
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

    def predict_attack_type(self, features: np.ndarray) -> Dict[str, Any]:
        """Thực hiện prediction attack type với CNN model."""
        try:
            # Get predictions
            predictions = self.model.predict(features, verbose=0)

            # Multi-class classification
            predicted_class_idx = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))

            # Get attack type name
            if hasattr(self.label_encoder, 'classes_'):
                predicted_attack_type = self.label_encoder.classes_[predicted_class_idx]
            else:
                predicted_attack_type = f"attack_type_{predicted_class_idx}"

            # Get top-k predictions
            top_k = 3
            top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]
            top_k_predictions = []

            for idx in top_k_indices:
                attack_type = self.label_encoder.classes_[idx] if hasattr(self.label_encoder, 'classes_') else f"attack_type_{idx}"
                prob = float(predictions[0][idx])
                top_k_predictions.append({
                    'attack_type': attack_type,
                    'probability': prob
                })

            result = {
                'predicted_attack_type': predicted_attack_type,
                'predicted_class_idx': predicted_class_idx,
                'confidence': confidence,
                'top_k_predictions': top_k_predictions,
                'all_probabilities': predictions[0].tolist(),
                'model_type': 'cnn_1d_level2_attack_types',
                'timestamp': time.time()
            }

            return self._convert_numpy_to_python_types(result)

        except Exception as e:
            self.logger.error("Error during attack type prediction: %s", str(e))
            return self._convert_numpy_to_python_types({
                'error': str(e),
                'model_type': 'cnn_1d_level2_attack_types',
                'timestamp': time.time()
            })

    def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Xử lý một message từ Kafka."""
        try:
            message_id = message.get('id', 'unknown')
            level1_prediction = message.get('level1_prediction', {})

            self.logger.debug("Processing message ID: %s", message_id)

            # Check if we should process this message
            if not self.should_process_message(level1_prediction):
                self.skipped_count += 1
                return None  # Skip benign traffic

            # Try to use preprocessed CNN features first (recommended)
            original_message = message.get('original_message', {})
            cnn_features = original_message.get('cnn_features')
            if cnn_features is not None:
                # Convert list back to numpy array if needed
                if isinstance(cnn_features, list):
                    X = np.array(cnn_features, dtype=np.float32)
                else:
                    X = np.array([cnn_features], dtype=np.float32)
                self.logger.debug("Using preprocessed CNN features for message %s", message_id)
            else:
                # Fallback: preprocess from raw features (not recommended - may cause inconsistencies)
                self.logger.warning("No preprocessed CNN features found, falling back to raw features preprocessing")
                original_features = original_message.get('features', {})
                X = self.preprocess_features(original_features)
                if X is None:
                    self.logger.warning("Failed to preprocess features for message %s", message_id)
                    self.error_count += 1
                    return None

            # Make attack type prediction
            prediction_result = self.predict_attack_type(X)

            # Combine với thông tin gốc
            result = {
                'id': message_id,
                'original_message': self._convert_numpy_to_python_types(message),  # Ensure native types
                'level2_prediction': self._convert_numpy_to_python_types(prediction_result),  # Ensure native types
                'processing_timestamp': time.time()
            }

            self.processed_count += 1
            self.attack_detected_count += 1

            # Log attack detection
            attack_type = prediction_result.get('predicted_attack_type', 'unknown')
            confidence = prediction_result.get('confidence', 0.0)
            predicted_class_idx = prediction_result.get('predicted_class_idx', 0)

            # Get readable attack type label
            readable_attack_type = self.attack_type_labels.get(attack_type, self.attack_type_labels.get(predicted_class_idx, attack_type))

            # Get top predictions for more context
            top_k = prediction_result.get('top_k_predictions', [])
            top_info = ""
            if top_k and len(top_k) > 1:
                alt_attack = top_k[1]['attack_type']
                alt_readable = self.attack_type_labels.get(alt_attack, alt_attack)
                top_info = f" | Top alternatives: {alt_readable} ({top_k[1]['probability']:.3f})"

            self.logger.info(f"Attack detected: {readable_attack_type} (class_idx: {predicted_class_idx}, confidence: {confidence:.3f}){top_info}")

            # Update attack types statistics (sử dụng readable labels)
            readable_attack_type = self.attack_type_labels.get(attack_type, self.attack_type_labels.get(predicted_class_idx, attack_type))
            if readable_attack_type not in self.attack_types_stats:
                self.attack_types_stats[readable_attack_type] = 0
            self.attack_types_stats[readable_attack_type] += 1
            if confidence > 0:
                self.attack_confidence_summary.setdefault(readable_attack_type, []).append(confidence)

            # Log stats periodically
            if self.processed_count % 10 == 0:
                self._log_summary()

            return result

        except Exception as e:
            self.error_count += 1
            self.logger.error("Error processing message: %s", str(e))
            return None

    def run(self) -> None:
        """Main service loop với logic reset offset tương tự các service khác."""
        self.logger.info("🚀 Starting Level 2 CNN Attack Types Prediction Service")
        self.logger.info("Input topic: %s", self.input_topic)
        self.logger.info("Output topic: %s", self.output_topic)
        self.logger.info("Confidence threshold: %.2f", self.confidence_threshold)

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
        self.attack_detected_count = 0
        self.logger.info("Reset statistics counters - starting fresh count")

        # Đọc toàn bộ backlog, không seek_to_end (đồng bộ RF logic)
        self.service_start_time = datetime.now()
        self.last_partition_assignment = set()
        # Reset confidence summary per run
        self.attack_confidence_summary = {}

        self.logger.info("Service started successfully, waiting for messages...")

        try:
            while True:
                # Kiểm tra nếu partition assignment thay đổi (rebalance) và tiếp tục đọc (không seek_to_end)
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

                                attack_type = result.get('level2_prediction', {}).get('predicted_attack_type', 'unknown')
                                predicted_class_idx = result.get('level2_prediction', {}).get('predicted_class_idx', 0)
                                readable_attack_type = self.attack_type_labels.get(attack_type, self.attack_type_labels.get(predicted_class_idx, attack_type))
                                self.logger.debug("Attack type prediction sent: %s", readable_attack_type)

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

    def _log_attack_summary(self) -> None:
        """Log summary of attack types detected."""
        if self.attack_types_stats:
            self.logger.info("Attack Types Summary:")
            total_attacks = sum(self.attack_types_stats.values())
            for attack_type, count in sorted(self.attack_types_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_attacks * 100) if total_attacks > 0 else 0
                self.logger.info(f"   {attack_type}: {count} attacks ({percentage:.1f}%)")
        else:
            self.logger.info("No attacks detected yet")

    def _log_summary(self) -> None:
        """Log summary statistics (RF-style) with counts, percentages, avg confidence."""
        self.logger.info("=" * 60)
        self.logger.info("LEVEL 2 PREDICTION SUMMARY (CNN - Attack Types):")
        self.logger.info("=" * 60)
        self.logger.info(f"Total predictions: {self.processed_count}")
        self.logger.info("")
        self.logger.info("Attack type distribution:")
        for label, count in sorted(self.attack_types_stats.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                percentage = (count / self.processed_count) * 100 if self.processed_count > 0 else 0
                conf_list = self.attack_confidence_summary.get(label, [])
                avg_conf = sum(conf_list) / len(conf_list) if conf_list else 0.0
                self.logger.info(f"  - {label}: {count} ({percentage:.1f}%) - Avg confidence: {avg_conf:.3f}")
        self.logger.info("=" * 60)

    def get_attack_summary(self) -> Dict[str, Any]:
        """Get detailed attack detection summary."""
        total_attacks = sum(self.attack_types_stats.values())
        elapsed = time.time() - self.start_time

        summary = {
            'total_attacks_detected': total_attacks,
            'attack_types_breakdown': {},
            'detection_rate_per_minute': total_attacks / (elapsed / 60) if elapsed > 0 else 0,
            'most_common_attack': None,
            'attack_diversity': len(self.attack_types_stats)
        }

        if self.attack_types_stats:
            # Calculate breakdown with percentages
            for attack_type, count in self.attack_types_stats.items():
                percentage = (count / total_attacks * 100) if total_attacks > 0 else 0
                summary['attack_types_breakdown'][attack_type] = {
                    'count': count,
                    'percentage': round(percentage, 2)
                }

            # Find most common attack
            most_common = max(self.attack_types_stats.items(), key=lambda x: x[1])
            summary['most_common_attack'] = {
                'type': most_common[0],
                'count': most_common[1],
                'percentage': round(most_common[1] / total_attacks * 100, 2) if total_attacks > 0 else 0
            }

        return summary

    def get_scanning_attacks_summary(self) -> Dict[str, Any]:
        """Get summary of scanning attacks specifically.

        Trong Level 2 CNN training, scanning attacks chủ yếu là:
        - PortScan (Class 2): Network port scanning attacks
        - Có thể bao gồm FTP-Patator, SSH-Patator nếu dataset có
        """
        scanning_keywords = ['scan', 'portscan', 'patator', 'brute']
        scanning_attacks = {}

        total_scanning = 0
        for attack_type, count in self.attack_types_stats.items():
            attack_lower = attack_type.lower()
            if any(keyword in attack_lower for keyword in scanning_keywords):
                scanning_attacks[attack_type] = count
                total_scanning += count

        elapsed = time.time() - self.start_time

        return {
            'total_scanning_attacks': total_scanning,
            'scanning_attack_types': scanning_attacks,
            'scanning_rate_per_minute': total_scanning / (elapsed / 60) if elapsed > 0 else 0,
            'scanning_percentage': (total_scanning / sum(self.attack_types_stats.values()) * 100) if self.attack_types_stats else 0,
            'most_common_scanning': max(scanning_attacks.items(), key=lambda x: x[1]) if scanning_attacks else None,
            'note': 'Level 2 CNN: PortScan (Class 2) is primary scanning attack detection'
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        elapsed = time.time() - self.start_time
        total_messages = self.processed_count + self.skipped_count + self.error_count
        return {
            'processed_attacks': self.processed_count,
            'skipped_benign': self.skipped_count,
            'error_count': self.error_count,
            'attack_detected_count': self.attack_detected_count,
            'attack_types_stats': dict(sorted(self.attack_types_stats.items(), key=lambda x: x[1], reverse=True)),
            'total_messages': total_messages,
            'uptime_seconds': elapsed,
            'processing_rate': self.processed_count / elapsed if elapsed > 0 else 0,
            'error_rate': self.error_count / total_messages if total_messages > 0 else 0,
            'attack_detection_rate': self.attack_detected_count / total_messages if total_messages > 0 else 0
        }

    def _log_final_summary(self):
        """Log summary statistics"""
        self.logger.info("=" * 60)
        self.logger.info("LEVEL 2 PREDICTION SUMMARY (CNN - Attack Types):")
        self.logger.info("=" * 60)
        self.logger.info(f"Total predictions: {self.processed_count}")

        if self.processed_count > 0:
            self.logger.info("")
            self.logger.info("Attack type distribution:")
            for attack_type, count in sorted(self.attack_types_stats.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    percentage = (count / self.processed_count * 100) if self.processed_count > 0 else 0
                    self.logger.info(f"  - {attack_type}: {count} ({percentage:.1f}%)")

        self.logger.info("=" * 60)

    def stop(self) -> None:
        """Stop service và cleanup resources."""
        # Log final summary
        self._log_final_summary()

        self.logger.info("🛑 Stopping Level 2 CNN Attack Types Prediction Service...")

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

    parser = argparse.ArgumentParser(description="Level 2 CNN Attack Types Prediction Service")
    parser.add_argument("--kafka-servers", default="localhost:9092", help="Kafka servers")
    parser.add_argument("--group-id", default="level2_cnn_service", help="Consumer group ID")
    parser.add_argument("--input-topic", default="level_1_predictions", help="Input topic")
    parser.add_argument("--output-topic", default="level_2_predictions", help="Output topic")
    parser.add_argument("--model-path", default="artifacts_cnn_level2/attack_classifier_cnn_final.h5", help="Path to CNN model")
    parser.add_argument("--scaler-path", default="artifacts_cnn_level2/scaler.joblib", help="Path to scaler")
    parser.add_argument("--label-encoder-path", default="artifacts_cnn_level2/label_encoder.joblib", help="Path to label encoder")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Confidence threshold for processing")
    parser.add_argument("--poll-timeout", type=int, default=1000, help="Poll timeout in ms")

    args = parser.parse_args()

    # Create and run service
    service = Level2CNNPredictionService(
        kafka_servers=args.kafka_servers,
        group_id=args.group_id,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        label_encoder_path=args.label_encoder_path,
        confidence_threshold=args.confidence_threshold,
        poll_timeout=args.poll_timeout
    )

    service.run()


if __name__ == "__main__":
    main()
