#!/usr/bin/env python3
"""
Safenet IDS - Level 1 Prediction Service
Dịch vụ chạy prediction Level 1 (phân loại nhóm attack tổng quát)

Chức năng chính:
- Đọc dữ liệu đã tiền xử lý từ Kafka topic 'preprocessed_events'
- Load và chạy model RandomForest Level 1 từ artifacts
- Phân loại thành 5 nhóm: benign, dos, ddos, bot, rare_attack
- Gửi kết quả prediction kèm confidence scores đến 'level1_predictions'

Model Level 1:
- Algorithm: RandomForestClassifier (ensemble learning)
- Input: 78+ network flow features đã được preprocessing
- Output: Encoded prediction (0-4) + confidence probabilities
- Training data: CICIDS2017 với hierarchical labels

Luồng xử lý:
1. Nhận message từ preprocessed_events
2. Load model và metadata nếu chưa có
3. Trích xuất features từ message
4. Chạy model.predict() và model.predict_proba()
5. Format kết quả và gửi đến level1_predictions
"""

# Import các thư viện cần thiết
import json  # Parse JSON messages và serialize results
import logging  # Comprehensive logging system
import joblib  # Load trained scikit-learn models
import numpy as np  # Numerical computations
import pandas as pd  # Data manipulation cho model input
from pathlib import Path  # File system path handling
from kafka import KafkaConsumer, KafkaProducer  # Kafka messaging
from datetime import datetime  # Timestamp generation

# Cấu hình logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('services/logs/level1_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Level1Prediction')

class Level1PredictionService:
    """Service chạy prediction Level 1 từ dữ liệu đã tiền xử lý"""

    def __init__(self,
                 kafka_bootstrap_servers='localhost:9092',
                 input_topic='preprocess_data',
                 output_topic='level_1_predictions',
                 model_path='artifacts/ids_pipeline.joblib',
                 metadata_path='artifacts/metadata.json'):
        """
        Khởi tạo Level 1 Prediction Service

        Args:
            kafka_bootstrap_servers: Kafka bootstrap servers
            input_topic: Topic để đọc dữ liệu đã xử lý
            output_topic: Topic để gửi kết quả prediction
            model_path: Đường dẫn đến model Level 1
            metadata_path: Đường dẫn đến metadata của model
        """
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.consumer = None
        self.producer = None
        self.kafka_servers = kafka_bootstrap_servers
        self.is_running = False

        # Model và metadata
        self.model = None
        self.metadata = None
        self.label_mapping = {
            0: 'benign',
            1: 'dos',
            2: 'ddos',
            3: 'bot',
            4: 'rare_attack'
        }

        # Khởi tạo model và Kafka clients
        self._load_model()
        self._load_metadata()
        self._init_consumer()
        self._init_producer()

    def _load_model(self):
        """Load model Level 1 từ file joblib"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model = joblib.load(self.model_path)
            logger.info(f"Loaded Level 1 model from {self.model_path}")

            # Test model với dummy data để đảm bảo hoạt động
            dummy_data = pd.DataFrame([{
                'timestamp': datetime.now().isoformat(),
                'source_ip': '192.168.1.1',
                'destination_ip': '192.168.1.2',
                'source_port': 12345,
                'destination_port': 80,
                'protocol': 6,
                'flow_duration': 1000000,
                'total_fwd_packets': 50,
                'total_backward_packets': 50,
                'total_length_of_fwd_packets': 5000,
                'total_length_of_bwd_packets': 5000,
                'fwd_packet_length_max': 1500,
                'fwd_packet_length_min': 60,
                'fwd_packet_length_mean': 500.0,
                'fwd_packet_length_std': 200.0,
                'bwd_packet_length_max': 1500,
                'bwd_packet_length_min': 60,
                'bwd_packet_length_mean': 500.0,
                'bwd_packet_length_std': 200.0,
                'flow_bytes_s': 1000000.0,
                'flow_packets_s': 10000.0,
                'flow_iat_mean': 10000.0,
                'flow_iat_std': 5000.0,
                'flow_iat_max': 100000.0,
                'flow_iat_min': 1000.0,
                'fwd_iat_total': 500000.0,
                'fwd_iat_mean': 10000.0,
                'fwd_iat_std': 5000.0,
                'fwd_iat_max': 100000.0,
                'fwd_iat_min': 1000.0,
                'bwd_iat_total': 500000.0,
                'bwd_iat_mean': 10000.0,
                'bwd_iat_std': 5000.0,
                'bwd_iat_max': 100000.0,
                'bwd_iat_min': 1000.0,
                'fwd_psh_flags': 1,
                'bwd_psh_flags': 1,
                'fwd_urg_flags': 0,
                'bwd_urg_flags': 0,
                'fwd_header_length': 60,
                'bwd_header_length': 60,
                'fwd_packets_s': 1000.0,
                'bwd_packets_s': 1000.0,
                'min_packet_length': 60,
                'max_packet_length': 1500,
                'packet_length_mean': 750.0,
                'packet_length_std': 300.0,
                'packet_length_variance': 90000.0,
                'fin_flag_count': 1,
                'syn_flag_count': 1,
                'rst_flag_count': 0,
                'psh_flag_count': 2,
                'ack_flag_count': 2,
                'urg_flag_count': 0,
                'cwe_flag_count': 0,
                'ece_flag_count': 0,
                'down_up_ratio': 1.0,
                'average_packet_size': 750.0,
                'avg_fwd_segment_size': 500.0,
                'avg_bwd_segment_size': 500.0,
                'fwd_header_length.1': 60,
                'fwd_avg_bytes_bulk': 0,
                'fwd_avg_packets_bulk': 0,
                'fwd_avg_bulk_rate': 0.0,
                'bwd_avg_bytes_bulk': 0,
                'bwd_avg_packets_bulk': 0,
                'bwd_avg_bulk_rate': 0.0,
                'subflow_fwd_packets': 25,
                'subflow_fwd_bytes': 2500,
                'subflow_bwd_packets': 25,
                'subflow_bwd_bytes': 2500,
                'init_win_bytes_forward': 65535,
                'init_win_bytes_backward': 65535,
                'act_data_pkt_fwd': 20,
                'min_seg_size_forward': 20,
                'active_mean': 10000.0,
                'active_std': 5000.0,
                'active_max': 100000.0,
                'active_min': 1000.0,
                'idle_mean': 10000.0,
                'idle_std': 5000.0,
                'idle_max': 100000.0,
                'idle_min': 1000.0,
                'label': 'Benign',
                'label_encoded': 0 # Added this for now
            }])
            test_pred = self.model.predict(dummy_data)
            logger.info(f"Model test prediction successful: {test_pred}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_metadata(self):
        """Load metadata của model"""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata from {self.metadata_path}")

                # Log thông tin model
                logger.info(f"Model type: {self.metadata.get('model_type', 'Unknown')}")
                logger.info(f"Training variant: {self.metadata.get('train_variant', 'Unknown')}")
                logger.info(f"Drop columns: {self.metadata.get('drop_columns_resolved', [])}")
            else:
                logger.warning(f"Metadata file not found: {self.metadata_path}, using defaults")

        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")

    def _init_consumer(self):
        """Khởi tạo Kafka consumer"""
        try:
            self.consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.kafka_servers,
                group_id='safenet-ids-level1-prediction-group',
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                key_deserializer=lambda x: x.decode('utf-8') if x else None,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                session_timeout_ms=30000,
                heartbeat_interval_ms=3000,
                max_poll_records=50
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

    def predict_single_record(self, record: dict) -> dict:
        """
        Chạy prediction Level 1 cho một network flow record

        Process chi tiết:
        1. Data Preparation:
           - Exclude non-feature columns (metadata, identifiers)
           - Filter chỉ numeric features (model chỉ work với numeric)
           - Validate có đủ features để predict

        2. Model Inference:
           - Chạy model.predict() -> encoded prediction (0-4)
           - Chạy model.predict_proba() -> confidence scores cho tất cả classes

        3. Result Formatting:
           - Map encoded prediction sang human-readable label
           - Structure kết quả theo schema chuẩn
           - Include debugging info (features used, model info)

        Args:
            record (dict): Network flow data đã được preprocessing

        Returns:
            dict: Structured prediction result với format:
            {
                'prediction_timestamp': ISO timestamp,
                'original_data': input record,
                'prediction': {
                    'encoded': 0-4,
                    'label': 'benign'|'dos'|etc,
                    'confidence': max probability,
                    'probabilities': {label: prob for each class}
                },
                'features_used': list of feature columns,
                'model_info': model metadata
            }
        """
        try:
            # ===== PHASE 1: Data Preparation =====
            # Loại bỏ các cột không phải features của model
            exclude_cols = [
                'timestamp', 'source_ip', 'destination_ip',  # Identifiers
                'label', 'label_group',  # Target variables (nếu có)
                'preprocessing_timestamp', 'preprocessing_metadata', 'preprocessing_error'  # Metadata
            ]

            # Convert dict sang DataFrame để dễ xử lý
            df = pd.DataFrame([record])

            # Lọc chỉ các cột numeric hợp lệ (model chỉ nhận numeric input)
            feature_cols = []
            for col in df.columns:
                if col not in exclude_cols:
                    # Kiểm tra data type và không phải NaN
                    if df[col].dtype in ['int64', 'float64', 'float32', 'int32']:
                        if not pd.isna(df[col].iloc[0]):
                            feature_cols.append(col)

            # Validate có đủ features
            if not feature_cols:
                raise ValueError("No valid numeric features found for Level 1 prediction")

            # Tạo DataFrame chỉ chứa features cho model
            features_df = df[feature_cols]

            # ===== PHASE 2: Model Inference =====
            # Chạy prediction - trả về encoded class (0-4)
            prediction_encoded = self.model.predict(features_df)[0]

            # Chạy predict probabilities - confidence scores cho tất cả classes
            prediction_proba = self.model.predict_proba(features_df)[0]

            # Map encoded prediction sang readable label
            prediction_label = self.label_mapping.get(int(prediction_encoded), 'unknown')

            # ===== PHASE 3: Result Formatting =====
            prediction_result = {
                # Metadata
                'prediction_timestamp': datetime.now().isoformat(),
                'original_data': record,  # Preserve original input

                # Prediction results
                'prediction': {
                    'encoded': int(prediction_encoded),  # 0-4 for model compatibility
                    'label': prediction_label,  # Human readable: benign, dos, etc.
                    'confidence': float(max(prediction_proba)),  # Max probability as confidence
                    'probabilities': {  # Full probability distribution
                        self.label_mapping[i]: float(prob)
                        for i, prob in enumerate(prediction_proba)
                    }
                },

                # Debugging info
                'features_used': feature_cols,  # Which features were actually used
                'model_info': {
                    'model_path': str(self.model_path),
                    'model_type': self.metadata.get('model_type', 'Unknown') if self.metadata else 'Unknown'
                }
            }

            return prediction_result

        except Exception as e:
            # Error handling - return structured error result
            logger.error(f"Error in Level 1 prediction: {e}")

            return {
                'prediction_timestamp': datetime.now().isoformat(),
                'original_data': record,
                'prediction_error': str(e),  # Error description
                'prediction': {
                    'encoded': -1,  # Invalid prediction
                    'label': 'error',  # Error state
                    'confidence': 0.0,  # No confidence
                    'probabilities': {}  # Empty probabilities
                }
            }

    def send_prediction_result(self, data: dict, original_key: str = None):
        """
        Gửi kết quả prediction đến Kafka topic

        Args:
            data: Kết quả prediction
            original_key: Key từ message gốc
        """
        try:
            key = original_key or data.get('prediction_timestamp', str(datetime.now().timestamp()))

            future = self.producer.send(self.output_topic, value=data, key=key)
            record_metadata = future.get(timeout=10)

            logger.info(f"Sent Level 1 prediction to {record_metadata.topic} "
                       f"partition {record_metadata.partition} "
                       f"offset {record_metadata.offset}")

        except Exception as e:
            logger.error(f"Failed to send prediction result: {e}")

    def start_prediction(self):
        """Bắt đầu quá trình prediction"""
        self.is_running = True
        logger.info(f"Starting Level 1 prediction service: {self.input_topic} -> {self.output_topic}")

        try:
            for message in self.consumer:
                if not self.is_running:
                    break

                try:
                    # Lấy dữ liệu từ message
                    processed_data = message.value
                    original_key = message.key

                    logger.info(f"Processing Level 1 prediction for key: {original_key}")

                    # Chạy prediction
                    prediction_result = self.predict_single_record(processed_data)

                    # Gửi kết quả
                    self.send_prediction_result(prediction_result, original_key)

                except Exception as e:
                    logger.error(f"Error processing prediction: {e}")
                    continue

        except KeyboardInterrupt:
            logger.info("Level 1 prediction service stopped by user")
        except Exception as e:
            logger.error(f"Error in prediction loop: {e}")
        finally:
            self.stop()

    def stop(self):
        """Dừng service"""
        self.is_running = False
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()
            logger.info("Level 1 prediction service stopped")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Safenet IDS - Level 1 Prediction Service')
    parser.add_argument('--kafka-servers', default='localhost:9092',
                       help='Kafka bootstrap servers')
    parser.add_argument('--input-topic', default='preprocess_data',
                       help='Input topic name')
    parser.add_argument('--output-topic', default='level_1_predictions',
                       help='Output topic name')
    parser.add_argument('--model-path', default='artifacts/ids_pipeline.joblib',
                       help='Path to Level 1 model file')
    parser.add_argument('--metadata-path', default='artifacts/metadata.json',
                       help='Path to model metadata file')

    args = parser.parse_args()

    # Tạo thư mục logs nếu chưa có
    import os
    os.makedirs('services/logs', exist_ok=True)

    # Khởi tạo và chạy service
    service = Level1PredictionService(
        kafka_bootstrap_servers=args.kafka_servers,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        model_path=args.model_path,
        metadata_path=args.metadata_path
    )

    try:
        logger.info("Starting Safenet IDS Level 1 Prediction Service...")
        service.start_prediction()
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service failed: {e}")


if __name__ == '__main__':
    main()
