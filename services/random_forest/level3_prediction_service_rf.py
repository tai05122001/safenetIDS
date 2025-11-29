#!/usr/bin/env python3
"""
Safenet IDS - Level 3 Prediction Service (Random Forest)
Đọc từ topic level_2_predictions, chạy model Level 3 (Random Forest) để phân loại chi tiết DoS.

Level 3: Phân loại chi tiết loại DoS (DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest)
Chỉ chạy khi Level 2 = dos

Model Level 3:
- Algorithm: RandomForestClassifier (multi-class classification)
- Input: Network flow features (loại bỏ label columns từ Level 1 và Level 2)
- Output: Encoded prediction (DoS subtypes) + confidence probabilities
- Training data: CICIDS2017 với label_encoded (DoS subtypes)
- Level description: DoS Detail (DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest)
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
        logging.FileHandler('services/logs/level3_prediction_rf.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Level3PredictionRF')

class Level3PredictionServiceRF:
    """Service chạy prediction Level 3 - DoS Detail (DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest) - sử dụng Random Forest model"""

    def __init__(self,
                 kafka_bootstrap_servers='localhost:9092',
                 input_topic='level_2_predictions',
                 output_topic='level_3_predictions',
                 model_path='artifacts_level3_dos_rf/dos/ids_pipeline_level3_dos_rf.joblib',
                 metadata_path='artifacts_level3_dos_rf/dos/metadata.json'):
        """
        Khởi tạo Level 3 Prediction Service (Random Forest)

        Args:
            kafka_bootstrap_servers: Kafka bootstrap servers
            input_topic: Topic để đọc kết quả Level 2
            output_topic: Topic để gửi kết quả Level 3
            model_path: Đường dẫn đến model Level 3 (Random Forest)
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
        # Model Level 3: DoS Detail classification
        self.label_mapping = {}
        # Feature names từ model pipeline
        self.expected_feature_names = None
        
        # Summary statistics
        self.prediction_count = 0
        self.prediction_summary = {
            'DoS Hulk': 0,
            'DoS GoldenEye': 0,
            'DoS slowloris': 0,
            'DoS Slowhttptest': 0,
            'error': 0
        }
        self.confidence_summary = {
            'DoS Hulk': [],
            'DoS GoldenEye': [],
            'DoS slowloris': [],
            'DoS Slowhttptest': []
        }

        # Khởi tạo model và Kafka clients
        self._load_model()
        self._load_metadata()
        self._init_consumer()
        self._init_producer()

    def _init_consumer(self):
        """Khởi tạo Kafka consumer"""
        try:
            self.consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.kafka_servers,
                group_id='safenet-ids-level3-prediction-rf-group',
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

    def _load_model(self):
        """Load model Level 3 từ file joblib"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model = joblib.load(self.model_path)
            logger.info(f"Loaded Level 3 Random Forest model from {self.model_path}")

            # Lấy feature names từ model pipeline
            try:
                preprocessor = self.model.named_steps.get('preprocess', None)
                if preprocessor is not None:
                    feature_names = []
                    for name, transformer, columns in preprocessor.transformers_:
                        if isinstance(columns, list):
                            feature_names.extend(columns)
                        elif hasattr(columns, '__iter__'):
                            feature_names.extend(list(columns))
                    
                    self.expected_feature_names = feature_names
                    logger.info(f"Extracted {len(feature_names)} expected feature names from model pipeline")
            except Exception as e:
                logger.warning(f"Could not extract feature names from model: {e}")

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
                logger.info(f"Level: {self.metadata.get('level', 'Unknown')}")
                logger.info(f"Level description: {self.metadata.get('level_description', 'Unknown')}")
                logger.info(f"Drop columns: {self.metadata.get('drop_columns_resolved', [])}")
                
                # Log label mapping từ metadata
                label_mapping = self.metadata.get('label_mapping', None)
                if label_mapping:
                    logger.info(f"Model label mapping: {label_mapping}")
                    # Cập nhật label_mapping từ metadata
                    self.label_mapping = {int(k): v for k, v in label_mapping.items()}
                else:
                    # Fallback: Default mapping cho DoS subtypes
                    self.label_mapping = {
                        2: 'DoS GoldenEye',
                        3: 'DoS Hulk',
                        4: 'DoS Slowhttptest',
                        5: 'DoS slowloris'
                    }
                    logger.warning("No label_mapping in metadata, using default DoS subtype mapping")
                
                # Log class weights nếu có
                class_weights = self.metadata.get('class_weights', None)
                if class_weights:
                    logger.info(f"Model class weights: {class_weights}")
            else:
                logger.warning(f"Metadata file not found: {self.metadata_path}, using defaults")
                # Fallback: Default mapping
                self.label_mapping = {
                    2: 'DoS GoldenEye',
                    3: 'DoS Hulk',
                    4: 'DoS Slowhttptest',
                    5: 'DoS slowloris'
                }

        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")

    def _should_run_level3(self, level2_result: Dict[str, Any]) -> bool:
        """
        Kiểm tra xem có cần chạy Level 3 prediction không

        Args:
            level2_result: Kết quả prediction Level 2

        Returns:
            True nếu Level 2 predict = dos, False nếu không
        """
        level2_prediction = level2_result.get('level2_prediction', {})
        label = level2_prediction.get('label', '')
        encoded = level2_prediction.get('encoded', -1)

        # Level 3 chỉ chạy khi Level 2 predict = dos (encoded=0)
        if label == 'dos' or encoded == 0:
            return True

        return False

    def predict_level3_single_record(self, level2_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Chạy prediction Level 3 - DoS Detail (DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest)

        Args:
            level2_result (Dict[str, Any]): Kết quả từ Level 2 prediction

        Returns:
            Dict[str, Any]: Structured Level 3 prediction result
        """
        try:
            # Lấy original_data từ level1_result (chứa features gốc)
            level1_result = level2_result.get('level1_result', {})
            original_data = level1_result.get('original_data', {})

            # Chuẩn bị dữ liệu cho prediction Level 3
            import pandas as pd
            import numpy as np

            # Loại bỏ các cột không phải features (giống hệt training)
            # Level 3: drop label_group, label, label_group_encoded, label_binary_encoded, label_attack_type_encoded
            exclude_cols = [
                'timestamp', 'source_ip', 'destination_ip',
                'label', 'label_group',
                'label_encoded', 'label_group_encoded',
                'label_binary_encoded',  # QUAN TRỌNG: Loại bỏ label column của Level 1
                'label_attack_type_encoded',  # QUAN TRỌNG: Loại bỏ label column của Level 2
                'preprocessing_timestamp', 'preprocessing_metadata', 'preprocessing_error'
            ]
            
            # Thêm các cột từ metadata nếu có
            if self.metadata and 'drop_columns_resolved' in self.metadata:
                metadata_drop_cols = self.metadata['drop_columns_resolved']
                for col in metadata_drop_cols:
                    if col not in exclude_cols:
                        exclude_cols.append(col)
                        logger.debug(f"Added drop column from metadata: {col}")

            df = pd.DataFrame([original_data])
            
            # Lọc chỉ các cột numeric hợp lệ (loại bỏ label columns)
            feature_cols = []
            for col in df.columns:
                if col not in exclude_cols:
                    if df[col].dtype in ['int64', 'float64', 'float32', 'int32']:
                        if not pd.isna(df[col].iloc[0]):
                            feature_cols.append(col)

            if not feature_cols:
                raise ValueError("No valid numeric features found for Level 3 prediction")

            # Feature alignment với model pipeline
            if self.expected_feature_names is not None:
                available_features = {col: df[col].iloc[0] for col in feature_cols if col in self.expected_feature_names}
                missing_features = [col for col in self.expected_feature_names if col not in available_features]
                
                if missing_features:
                    logger.warning(f"Missing {len(missing_features)} expected features: {missing_features[:10]}...")
                    for col in missing_features:
                        available_features[col] = 0.0
                
                predict_df = pd.DataFrame([available_features])[self.expected_feature_names]
                logger.info(f"Using {len(self.expected_feature_names)} features (aligned with model pipeline)")
            else:
                predict_df = df[feature_cols]
                logger.warning(f"Using {len(feature_cols)} features (model pipeline feature names not available)")

            # Chạy prediction
            try:
                prediction_encoded = self.model.predict(predict_df)[0]
                prediction_proba = self.model.predict_proba(predict_df)[0]
            except (ValueError, KeyError) as e:
                logger.error(f"Level 3 prediction failed: {e}")
                logger.error(f"Expected features: {self.expected_feature_names[:20] if self.expected_feature_names else 'N/A'}...")
                logger.error(f"Available features: {list(predict_df.columns)[:20]}...")
                raise

            # Map encoded prediction sang readable label (DoS subtypes)
            prediction_encoded_int = int(prediction_encoded)
            prediction_label = self.label_mapping.get(prediction_encoded_int, f'DoS_unknown_{prediction_encoded_int}')
            
            # Log probabilities
            probabilities_dict = {}
            for i, prob in enumerate(prediction_proba):
                label_name = self.label_mapping.get(i, f'DoS_type_{i}')
                probabilities_dict[label_name] = float(prob)
            
            prob_str = ", ".join([f"{label}: {prob:.3f}" for label, prob in sorted(probabilities_dict.items())])
            logger.info(f"Level 3 prediction probabilities: {prob_str}")
            logger.info(f"DoS subtype prediction: {prediction_encoded_int} -> {prediction_label}")

            # Tạo kết quả Level 3
            level3_result = {
                'level3_prediction_timestamp': datetime.now().isoformat(),
                'level2_result': level2_result,
                'level3_prediction': {
                    'encoded': prediction_encoded_int,
                    'label': prediction_label,
                    'confidence': float(max(prediction_proba)),
                    'probabilities': probabilities_dict
                },
                'features_used': list(predict_df.columns) if hasattr(predict_df, 'columns') else feature_cols,
                'model_info': {
                    'model_path': str(self.model_path),
                    'model_type': self.metadata.get('model_type', 'random_forest') if self.metadata else 'random_forest'
                }
            }

            return level3_result

        except Exception as e:
            logger.error(f"Error in Level 3 prediction: {e}")
            # Trả về error result
            return {
                'level3_prediction_timestamp': datetime.now().isoformat(),
                'level2_result': level2_result,
                'level3_prediction_error': str(e),
                'level3_prediction': {
                    'encoded': -1,
                    'label': 'error',
                    'confidence': 0.0,
                    'probabilities': {}
                }
            }

    def send_level3_result(self, data: Dict[str, Any], original_key: str = None):
        """Gửi kết quả Level 3 prediction đến Kafka topic"""
        try:
            key = original_key or data.get('level3_prediction_timestamp', str(datetime.now().timestamp()))

            future = self.producer.send(self.output_topic, value=data, key=key)
            record_metadata = future.get(timeout=10)

            logger.info(f"Sent Level 3 prediction to {record_metadata.topic} "
                       f"partition {record_metadata.partition} "
                       f"offset {record_metadata.offset}")

        except Exception as e:
            logger.error(f"Failed to send Level 3 prediction result: {e}")

    def process_level2_result(self, level2_result: Dict[str, Any], original_key: str = None):
        """Xử lý kết quả từ Level 2 prediction"""
        try:
            # Kiểm tra xem có cần chạy Level 3 không (chỉ khi Level 2 = dos)
            should_run = self._should_run_level3(level2_result)

            if should_run:
                logger.info("Running Level 3 prediction (DoS detail: DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest)")

                # Chạy Level 3 prediction
                level3_result = self.predict_level3_single_record(level2_result)

                # Cập nhật summary statistics
                self._update_summary(level3_result)

                # Gửi kết quả Level 3
                self.send_level3_result(level3_result, original_key)
                
                # Log summary mỗi 10 predictions
                if self.prediction_count % 10 == 0:
                    self._log_summary()
            else:
                # Level 2 đã xác định ddos hoặc portscan - không cần Level 3
                level2_prediction = level2_result.get('level2_prediction', {})
                label = level2_prediction.get('label', '')

                logger.info(f"Skipping Level 3 prediction for non-DoS attack: {label}")

        except Exception as e:
            logger.error(f"Error processing Level 2 result: {e}")

    def start_prediction(self):
        """Bắt đầu quá trình prediction Level 3"""
        self.is_running = True
        logger.info(f"Starting Level 3 prediction service (RF): {self.input_topic} -> {self.output_topic}")
        logger.info(f"Model loaded: {self.model_path}")

        try:
            for message in self.consumer:
                if not self.is_running:
                    break

                try:
                    level2_result = message.value
                    original_key = message.key

                    logger.info(f"Processing Level 3 prediction for key: {original_key}")

                    self.process_level2_result(level2_result, original_key)

                except Exception as e:
                    logger.error(f"Error processing Level 3 prediction: {e}")
                    continue

        except KeyboardInterrupt:
            logger.info("Level 3 prediction service stopped by user")
        except Exception as e:
            logger.error(f"Error in Level 3 prediction loop: {e}")
        finally:
            self.stop()

    def _update_summary(self, level3_result: Dict[str, Any]):
        """Cập nhật summary statistics"""
        self.prediction_count += 1
        
        level3_pred = level3_result.get('level3_prediction', {})
        label = level3_pred.get('label', 'error')
        confidence = level3_pred.get('confidence', 0.0)
        
        if label in self.prediction_summary:
            self.prediction_summary[label] += 1
            if label != 'error' and confidence > 0:
                self.confidence_summary[label].append(confidence)
    
    def _log_summary(self):
        """Log summary statistics"""
        logger.info("=" * 60)
        logger.info("LEVEL 3 PREDICTION SUMMARY (Random Forest - DoS Detail):")
        logger.info("=" * 60)
        logger.info(f"Total predictions: {self.prediction_count}")
        logger.info("")
        logger.info("DoS subtype distribution:")
        for label, count in sorted(self.prediction_summary.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                percentage = (count / self.prediction_count) * 100 if self.prediction_count > 0 else 0
                avg_conf = 0.0
                if label in self.confidence_summary and len(self.confidence_summary[label]) > 0:
                    avg_conf = sum(self.confidence_summary[label]) / len(self.confidence_summary[label])
                logger.info(f"  - {label}: {count} ({percentage:.1f}%) - Avg confidence: {avg_conf:.3f}")
        logger.info("=" * 60)
    
    def stop(self):
        """Dừng service"""
        self.is_running = False
        
        # Log final summary
        if self.prediction_count > 0:
            logger.info("")
            self._log_summary()
        
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()
            logger.info("Level 3 prediction service (RF) stopped")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Safenet IDS - Level 3 Prediction Service (Random Forest)')
    parser.add_argument('--kafka-servers', default='localhost:9092',
                       help='Kafka bootstrap servers')
    parser.add_argument('--input-topic', default='level_2_predictions',
                       help='Input topic name (Level 2 predictions)')
    parser.add_argument('--output-topic', default='level_3_predictions',
                       help='Output topic name (Level 3 predictions)')
    parser.add_argument('--model-path', default='artifacts_level3_dos_rf/dos/ids_pipeline_level3_dos_rf.joblib',
                       help='Path to Level 3 Random Forest model file')
    parser.add_argument('--metadata-path', default='artifacts_level3_dos_rf/dos/metadata.json',
                       help='Path to model metadata file')

    args = parser.parse_args()

    # Tạo thư mục logs nếu chưa có
    import os
    os.makedirs('services/logs', exist_ok=True)

    # Khởi tạo và chạy service
    service = Level3PredictionServiceRF(
        kafka_bootstrap_servers=args.kafka_servers,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        model_path=args.model_path,
        metadata_path=args.metadata_path
    )

    try:
        logger.info("Starting Safenet IDS Level 3 Prediction Service (Random Forest)...")
        service.start_prediction()
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service failed: {e}")


if __name__ == '__main__':
    main()

