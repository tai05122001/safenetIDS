#!/usr/bin/env python3
"""
Safenet IDS - Level 2 Prediction Service
Đọc từ topic level1_predictions, chạy model Level 2 cho các nhóm dos/rare_attack,
gửi đến topic level2_predictions
"""

import json
import logging
import joblib
import os
from pathlib import Path
from kafka import KafkaConsumer, KafkaProducer
from datetime import datetime
from typing import Dict, Any, Optional

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('services/logs/level2_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Level2Prediction')

class Level2PredictionService:
    """Service chạy prediction Level 2 cho các nhóm dos và rare_attack"""

    def __init__(self,
                 kafka_bootstrap_servers='localhost:9092',
                 input_topic='level1_predictions',
                 output_topic='level2_predictions',
                 models_base_path='artifacts_level2'):
        """
        Khởi tạo Level 2 Prediction Service

        Args:
            kafka_bootstrap_servers: Kafka bootstrap servers
            input_topic: Topic để đọc kết quả Level 1
            output_topic: Topic để gửi kết quả Level 2
            models_base_path: Thư mục chứa các model Level 2
        """
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.models_base_path = Path(models_base_path)
        self.consumer = None
        self.producer = None
        self.kafka_servers = kafka_bootstrap_servers
        self.is_running = False

        # Models Level 2 theo nhóm
        self.level2_models = {}
        self.level2_metadata = {}

        # Các nhóm cần Level 2 prediction
        self.level2_groups = ['dos', 'rare_attack']

        # Khởi tạo Kafka clients và load models
        self._init_consumer()
        self._init_producer()
        self._load_level2_models()

    def _init_consumer(self):
        """Khởi tạo Kafka consumer"""
        try:
            self.consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.kafka_servers,
                group_id='safenet-ids-level2-prediction-group',
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

    def _load_level2_models(self):
        """Load các model Level 2 cho từng nhóm"""
        for group in self.level2_groups:
            model_path = self.models_base_path / group / f"{group}_pipeline.joblib"
            metadata_path = self.models_base_path / group / "metadata.json"

            try:
                if model_path.exists():
                    self.level2_models[group] = joblib.load(model_path)
                    logger.info(f"Loaded Level 2 model for group '{group}' from {model_path}")

                    # Test model
                    import pandas as pd
                    dummy_data = pd.DataFrame([{
                        'flow_duration': 1000,
                        'total_fwd_packets': 10,
                        'total_backward_packets': 10
                    }])
                    test_pred = self.level2_models[group].predict(dummy_data)
                    logger.info(f"Model '{group}' test prediction successful: {test_pred}")
                else:
                    logger.warning(f"Level 2 model for group '{group}' not found at {model_path}")

                # Load metadata nếu có
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        self.level2_metadata[group] = json.load(f)
                    logger.info(f"Loaded metadata for group '{group}' from {metadata_path}")
                else:
                    logger.warning(f"Metadata for group '{group}' not found at {metadata_path}")

            except Exception as e:
                logger.error(f"Failed to load model for group '{group}': {e}")
                continue

        logger.info(f"Loaded {len(self.level2_models)} Level 2 models out of {len(self.level2_groups)} groups")

    def _should_run_level2(self, level1_prediction: Dict[str, Any]) -> Optional[str]:
        """
        Kiểm tra xem có cần chạy Level 2 prediction không

        Args:
            level1_prediction: Kết quả prediction Level 1

        Returns:
            Tên nhóm cần chạy Level 2, hoặc None nếu không cần
        """
        prediction = level1_prediction.get('prediction', {})
        label = prediction.get('label', '')

        if label in self.level2_groups and label in self.level2_models:
            return label

        return None

    def predict_level2_single_record(self, level1_result: Dict[str, Any], group: str) -> Dict[str, Any]:
        """
        Chạy prediction Level 2 cho nhóm attack cụ thể (dos hoặc rare_attack)

        Process chi tiết:
        1. Model Selection:
           - Chọn model tương ứng với group (dos/rare_attack)
           - Validate model đã được load

        2. Feature Extraction:
           - Sử dụng lại logic preprocessing từ Level 1
           - Extract numeric features từ original network data
           - Exclude metadata và identifier columns

        3. Model Inference:
           - Chạy model.predict() cho classification
           - Chạy model.predict_proba() cho confidence scores
           - Map prediction sang attack type cụ thể

        4. Result Integration:
           - Kết hợp kết quả Level 1 + Level 2
           - Thêm detailed attack classification
           - Include confidence và debugging info

        Args:
            level1_result (Dict[str, Any]): Kết quả từ Level 1 prediction
            group (str): Nhóm attack cần detailed classification ('dos' hoặc 'rare_attack')

        Returns:
            Dict[str, Any]: Structured Level 2 prediction result
        """
        try:
            if group not in self.level2_models:
                raise ValueError(f"No model available for group: {group}")

            model = self.level2_models[group]
            original_data = level1_result.get('original_data', {})

            # Chuẩn bị dữ liệu cho prediction Level 2
            # Sử dụng lại logic preprocessing từ Level 1
            import pandas as pd

            exclude_cols = [
                'timestamp', 'source_ip', 'destination_ip', 'label', 'label_group',
                'preprocessing_timestamp', 'preprocessing_metadata', 'preprocessing_error'
            ]

            df = pd.DataFrame([original_data])
            feature_cols = []
            for col in df.columns:
                if col not in exclude_cols and df[col].dtype in ['int64', 'float64', 'float32', 'int32']:
                    if not pd.isna(df[col].iloc[0]):
                        feature_cols.append(col)

            if not feature_cols:
                raise ValueError("No valid numeric features found for Level 2 prediction")

            features_df = df[feature_cols]

            # Chạy prediction
            prediction_encoded = model.predict(features_df)[0]
            prediction_proba = model.predict_proba(features_df)[0]

            # Mapping prediction sang label cụ thể theo group
            # Trong thực tế, cần mapping dựa trên metadata của model Level 2
            if group == 'dos':
                # Ví dụ mapping cho DoS attacks
                dos_labels = {
                    0: 'DoS Hulk',
                    1: 'DoS GoldenEye',
                    2: 'DoS slowloris',
                    3: 'DoS Slowhttptest',
                    4: 'Heartbleed'
                }
                prediction_label = dos_labels.get(int(prediction_encoded), f'DoS_unknown_{prediction_encoded}')
            elif group == 'rare_attack':
                # Ví dụ mapping cho rare attacks
                rare_labels = {
                    0: 'Web Attack',
                    1: 'Brute Force',
                    2: 'XSS',
                    3: 'SQL Injection',
                    4: 'Infiltration',
                    5: 'FTP-Patator',
                    6: 'SSH-Patator'
                }
                prediction_label = rare_labels.get(int(prediction_encoded), f'Rare_unknown_{prediction_encoded}')
            else:
                prediction_label = f'{group}_unknown_{prediction_encoded}'

            # Tạo kết quả Level 2
            level2_result = {
                'level2_prediction_timestamp': datetime.now().isoformat(),
                'level1_result': level1_result,
                'level2_prediction': {
                    'group': group,
                    'encoded': int(prediction_encoded),
                    'label': prediction_label,
                    'confidence': float(max(prediction_proba)),
                    'probabilities': {
                        str(i): float(prob) for i, prob in enumerate(prediction_proba)
                    }
                },
                'features_used': feature_cols,
                'model_info': {
                    'model_path': str(self.models_base_path / group / f"{group}_pipeline.joblib"),
                    'model_type': self.level2_metadata.get(group, {}).get('model_type', 'Unknown')
                }
            }

            return level2_result

        except Exception as e:
            logger.error(f"Error in Level 2 prediction for group {group}: {e}")
            # Trả về error result
            return {
                'level2_prediction_timestamp': datetime.now().isoformat(),
                'level1_result': level1_result,
                'level2_prediction_error': str(e),
                'level2_prediction': {
                    'group': group,
                    'encoded': -1,
                    'label': f'{group}_error',
                    'confidence': 0.0,
                    'probabilities': {}
                }
            }

    def send_level2_result(self, data: Dict[str, Any], original_key: str = None):
        """
        Gửi kết quả Level 2 prediction đến Kafka topic

        Args:
            data: Kết quả Level 2 prediction
            original_key: Key từ message gốc
        """
        try:
            key = original_key or data.get('level2_prediction_timestamp', str(datetime.now().timestamp()))

            future = self.producer.send(self.output_topic, value=data, key=key)
            record_metadata = future.get(timeout=10)

            logger.info(f"Sent Level 2 prediction to {record_metadata.topic} "
                       f"partition {record_metadata.partition} "
                       f"offset {record_metadata.offset}")

        except Exception as e:
            logger.error(f"Failed to send Level 2 prediction result: {e}")

    def process_level1_result(self, level1_result: Dict[str, Any], original_key: str = None):
        """
        Xử lý kết quả từ Level 1 prediction

        Args:
            level1_result: Kết quả từ Level 1
            original_key: Key của message gốc
        """
        try:
            # Kiểm tra xem có cần chạy Level 2 không
            group_to_predict = self._should_run_level2(level1_result)

            if group_to_predict:
                logger.info(f"Running Level 2 prediction for group: {group_to_predict}")

                # Chạy Level 2 prediction
                level2_result = self.predict_level2_single_record(level1_result, group_to_predict)

                # Gửi kết quả Level 2
                self.send_level2_result(level2_result, original_key)
            else:
                # Level 1 đã xác định benign, ddos, hoặc bot - không cần Level 2
                level1_prediction = level1_result.get('prediction', {})
                label = level1_prediction.get('label', '')

                logger.info(f"Skipping Level 2 prediction for benign/final prediction: {label}")

                # Có thể gửi đến topic alerts luôn, hoặc chỉ log
                # Ở đây chúng ta chỉ log vì alerting service sẽ xử lý

        except Exception as e:
            logger.error(f"Error processing Level 1 result: {e}")

    def start_prediction(self):
        """Bắt đầu quá trình prediction Level 2"""
        self.is_running = True
        logger.info(f"Starting Level 2 prediction service: {self.input_topic} -> {self.output_topic}")
        logger.info(f"Available Level 2 models: {list(self.level2_models.keys())}")

        try:
            for message in self.consumer:
                if not self.is_running:
                    break

                try:
                    # Lấy dữ liệu từ message
                    level1_result = message.value
                    original_key = message.key

                    logger.info(f"Processing Level 2 prediction for key: {original_key}")

                    # Xử lý kết quả Level 1
                    self.process_level1_result(level1_result, original_key)

                except Exception as e:
                    logger.error(f"Error processing Level 2 prediction: {e}")
                    continue

        except KeyboardInterrupt:
            logger.info("Level 2 prediction service stopped by user")
        except Exception as e:
            logger.error(f"Error in Level 2 prediction loop: {e}")
        finally:
            self.stop()

    def stop(self):
        """Dừng service"""
        self.is_running = False
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()
            logger.info("Level 2 prediction service stopped")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Safenet IDS - Level 2 Prediction Service')
    parser.add_argument('--kafka-servers', default='localhost:9092',
                       help='Kafka bootstrap servers')
    parser.add_argument('--input-topic', default='level1_predictions',
                       help='Input topic name (Level 1 predictions)')
    parser.add_argument('--output-topic', default='level2_predictions',
                       help='Output topic name (Level 2 predictions)')
    parser.add_argument('--models-base-path', default='artifacts_level2',
                       help='Base path for Level 2 models')

    args = parser.parse_args()

    # Tạo thư mục logs nếu chưa có
    import os
    os.makedirs('services/logs', exist_ok=True)

    # Khởi tạo và chạy service
    service = Level2PredictionService(
        kafka_bootstrap_servers=args.kafka_servers,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        models_base_path=args.models_base_path
    )

    try:
        logger.info("Starting Safenet IDS Level 2 Prediction Service...")
        service.start_prediction()
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service failed: {e}")


if __name__ == '__main__':
    main()
