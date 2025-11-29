#!/usr/bin/env python3
"""
Safenet IDS - Level 1 Prediction Service (Random Forest)
Dịch vụ chạy prediction Level 1 sử dụng Random Forest model từ artifacts_rf

Chức năng chính:
- Đọc dữ liệu đã tiền xử lý từ Kafka topic 'preprocess_data'
- Load và chạy model RandomForest Level 1 từ artifacts_rf
- Binary classification: Phân loại benign (0) vs attack (1)
- Gửi kết quả prediction kèm confidence scores đến 'level_1_predictions'

Model Level 1:
- Algorithm: RandomForestClassifier (binary classification)
- Input: Network flow features đã được preprocessing (loại bỏ label columns)
- Output: Encoded prediction (0=benign, 1=attack) + confidence probabilities
- Training data: CICIDS2017 với binary labels (label_binary_encoded)
- Level description: Binary Classification (benign vs attack)

Luồng xử lý:
1. Nhận message từ preprocess_data
2. Load model và metadata nếu chưa có
3. Trích xuất features từ message
4. Chạy model.predict() và model.predict_proba()
5. Format kết quả và gửi đến level_1_predictions
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
        logging.FileHandler('services/logs/level1_prediction_rf.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Level1PredictionRF')

class Level1PredictionServiceRF:
    """Service chạy prediction Level 1 từ dữ liệu đã tiền xử lý - sử dụng Random Forest model"""

    def __init__(self,
                 kafka_bootstrap_servers='localhost:9092',
                 input_topic='preprocess_data',
                 output_topic='level_1_predictions',
                 model_path='artifacts_rf/ids_pipeline_rf.joblib',
                 metadata_path='artifacts_rf/metadata.json'):
        """
        Khởi tạo Level 1 Prediction Service (Random Forest)

        Args:
            kafka_bootstrap_servers: Kafka bootstrap servers
            input_topic: Topic để đọc dữ liệu đã xử lý
            output_topic: Topic để gửi kết quả prediction
            model_path: Đường dẫn đến model Level 1 (Random Forest)
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
        # Model Level 1: Binary classification (0=benign, 1=attack)
        self.label_mapping = {
            0: 'benign',
            1: 'attack'
        }
        # Feature names từ model pipeline (để đảm bảo đúng thứ tự và tên cột)
        self.expected_feature_names = None
        
        # Summary statistics
        self.prediction_count = 0
        self.prediction_summary = {
            'benign': 0,
            'attack': 0,
            'error': 0
        }
        self.confidence_summary = {
            'benign': [],
            'attack': []
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
            logger.info(f"Loaded Level 1 Random Forest model from {self.model_path}")

            # Lấy feature names từ model pipeline để đảm bảo đúng thứ tự và tên cột
            try:
                preprocessor = self.model.named_steps.get('preprocess', None)
                if preprocessor is not None:
                    # Lấy tất cả feature names từ ColumnTransformer
                    feature_names = []
                    for name, transformer, columns in preprocessor.transformers_:
                        if isinstance(columns, list):
                            feature_names.extend(columns)
                        elif hasattr(columns, '__iter__'):
                            feature_names.extend(list(columns))
                    
                    # Lưu expected feature names để sử dụng khi predict
                    self.expected_feature_names = feature_names
                    logger.info(f"Extracted {len(feature_names)} expected feature names from model pipeline")
                    logger.debug(f"Expected features (first 10): {feature_names[:10]}")
                    
                    # Test model với dummy data để đảm bảo hoạt động
                    dummy_data_dict = {col: 0.0 for col in feature_names}
                    for col in dummy_data_dict:
                        if 'duration' in col.lower() or 'total' in col.lower():
                            dummy_data_dict[col] = 1000.0
                        elif 'mean' in col.lower() or 'avg' in col.lower():
                            dummy_data_dict[col] = 500.0
                        elif 'max' in col.lower():
                            dummy_data_dict[col] = 1500.0
                        elif 'min' in col.lower():
                            dummy_data_dict[col] = 60.0
                        elif 'count' in col.lower() or 'flags' in col.lower():
                            dummy_data_dict[col] = 1
                        else:
                            dummy_data_dict[col] = 0.0
                    
                    dummy_data = pd.DataFrame([dummy_data_dict])
                    test_pred = self.model.predict(dummy_data)
                    logger.info(f"Model test prediction successful: {test_pred}")
                else:
                    logger.warning("Could not extract feature names from model, skipping test")
            except Exception as e:
                logger.warning(f"Model test failed (non-critical): {e}. Model loaded successfully but test skipped.")

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
                logger.info(f"Training variant: {self.metadata.get('train_variant', 'Unknown')}")
                logger.info(f"Drop columns: {self.metadata.get('drop_columns_resolved', [])}")
                
                # Log label mapping từ metadata
                label_mapping = self.metadata.get('label_mapping', None)
                if label_mapping:
                    logger.info(f"Model label mapping: {label_mapping}")
                    # Cập nhật label_mapping từ metadata nếu có
                    self.label_mapping = {int(k): v for k, v in label_mapping.items()}
                
                # Log class weights nếu có (quan trọng để debug bias)
                class_weights = self.metadata.get('class_weights', None)
                if class_weights:
                    logger.info(f"Model class weights: {class_weights}")
                else:
                    logger.warning("⚠️  No class_weights found in metadata - model may have been trained without custom weights!")
                
                # Log class distribution nếu có
                class_dist = self.metadata.get('class_distribution', None)
                if class_dist:
                    logger.info(f"Training class distribution: {class_dist}")
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
                group_id='safenet-ids-level1-prediction-rf-group',
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

        Args:
            record (dict): Network flow data đã được preprocessing

        Returns:
            dict: Structured prediction result
        """
        try:
            # ===== PHASE 1: Data Preparation =====
            # Loại bỏ các cột không phải features (giống hệt training)
            # Level 1 binary classification: drop label_group, label, label_group_encoded, label_attack_type_encoded
            exclude_cols = [
                'timestamp', 'source_ip', 'destination_ip',
                'label', 'label_group',
                'label_encoded', 'label_group_encoded',
                'label_binary_encoded',  # QUAN TRỌNG: Loại bỏ label column dùng cho training
                'label_attack_type_encoded',  # QUAN TRỌNG: Loại bỏ label column của level 2
                'preprocessing_timestamp', 'preprocessing_metadata', 'preprocessing_error'
            ]
            
            # Thêm các cột từ metadata nếu có (các cột đã được drop khi training)
            if self.metadata and 'drop_columns_resolved' in self.metadata:
                metadata_drop_cols = self.metadata['drop_columns_resolved']
                for col in metadata_drop_cols:
                    if col not in exclude_cols:
                        exclude_cols.append(col)
                        logger.debug(f"Added drop column from metadata: {col}")

            # Convert dict sang DataFrame
            df = pd.DataFrame([record])

            # Lọc chỉ các cột numeric hợp lệ (loại bỏ label columns)
            feature_cols = []
            for col in df.columns:
                if col not in exclude_cols:
                    if df[col].dtype in ['int64', 'float64', 'float32', 'int32']:
                        if not pd.isna(df[col].iloc[0]):
                            feature_cols.append(col)

            if not feature_cols:
                raise ValueError("No valid numeric features found for Level 1 prediction")

            # ===== PHASE 2: Feature Alignment =====
            # Đảm bảo features được extract đúng thứ tự và tên cột giống với training
            if self.expected_feature_names is not None:
                # Sử dụng expected feature names từ model pipeline
                # Đảm bảo thứ tự và tên cột khớp với training
                available_features = {col: df[col].iloc[0] for col in feature_cols if col in self.expected_feature_names}
                missing_features = [col for col in self.expected_feature_names if col not in available_features]
                
                if missing_features:
                    logger.warning(f"Missing {len(missing_features)} expected features: {missing_features[:10]}...")
                    # Điền 0 cho các features thiếu (model sẽ xử lý qua imputer)
                    for col in missing_features:
                        available_features[col] = 0.0
                
                # Tạo DataFrame với đúng thứ tự features như training
                predict_df = pd.DataFrame([available_features])[self.expected_feature_names]
                logger.info(f"Using {len(self.expected_feature_names)} features (aligned with model pipeline)")
            else:
                # Fallback: Sử dụng features tìm được (không có expected names)
                predict_df = df[feature_cols]
                logger.warning(f"Using {len(feature_cols)} features (model pipeline feature names not available)")
                logger.warning("Feature alignment may not match training - this may cause prediction errors!")

            # ===== PHASE 3: Model Inference =====
            # Model pipeline đã có StandardScaler, không cần scale trước
            # Chỉ cần đảm bảo features đúng thứ tự và tên cột
            try:
                prediction_encoded = self.model.predict(predict_df)[0]
            except (ValueError, KeyError) as e:
                error_msg = str(e)
                logger.error(f"Model prediction failed: {error_msg}")
                if 'columns are missing' in error_msg or 'feature' in error_msg.lower():
                    logger.error("Feature mismatch detected! Check feature names and order.")
                    logger.error(f"Expected features: {self.expected_feature_names[:20] if self.expected_feature_names else 'N/A'}...")
                    logger.error(f"Available features: {list(predict_df.columns)[:20]}...")
                raise

            # Chạy predict probabilities TRƯỚC khi log prediction
            try:
                prediction_proba = self.model.predict_proba(predict_df)[0]
            except (ValueError, KeyError) as e:
                logger.error(f"predict_proba() failed: {e}")
                raise

            # Log chi tiết probabilities để debug
            probabilities_dict = {}
            for i, prob in enumerate(prediction_proba):
                label_name = self.label_mapping.get(i, f'class_{i}')
                probabilities_dict[label_name] = float(prob)
            
            # Log probabilities cho binary classification
            prob_str = ", ".join([f"{label}: {prob:.3f}" for label, prob in sorted(probabilities_dict.items())])
            logger.info(f"Prediction probabilities: {prob_str}")
            
            # Map encoded prediction sang readable label (binary: 0=benign, 1=attack)
            prediction_encoded_int = int(prediction_encoded)
            prediction_label = self.label_mapping.get(prediction_encoded_int, 'unknown')
            
            logger.info(f"Binary prediction: {prediction_encoded_int} -> {prediction_label}")

            # ===== PHASE 3: Result Formatting =====
            prediction_result = {
                'prediction_timestamp': datetime.now().isoformat(),
                'original_data': record,
                'prediction': {
                    'encoded': prediction_encoded_int,  # Sử dụng encoded đã được remap nếu cần
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

            return prediction_result

        except Exception as e:
            logger.error(f"Error in Level 1 prediction: {e}")

            return {
                'prediction_timestamp': datetime.now().isoformat(),
                'original_data': record,
                'prediction_error': str(e),
                'prediction': {
                    'encoded': -1,
                    'label': 'error',
                    'confidence': 0.0,
                    'probabilities': {}
                }
            }

    def send_prediction_result(self, data: dict, original_key: str = None):
        """Gửi kết quả prediction đến Kafka topic"""
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
        logger.info(f"Starting Level 1 prediction service (RF): {self.input_topic} -> {self.output_topic}")

        try:
            for message in self.consumer:
                if not self.is_running:
                    break

                try:
                    processed_data = message.value
                    original_key = message.key

                    logger.info(f"Processing Level 1 prediction for key: {original_key}")

                    prediction_result = self.predict_single_record(processed_data)

                    self._update_summary(prediction_result)

                    prediction = prediction_result.get('prediction', {})
                    label = prediction.get('label', 'unknown')
                    confidence = prediction.get('confidence', 0.0)
                    encoded = prediction.get('encoded', -1)
                    probabilities = prediction.get('probabilities', {})
                    
                    # Log chi tiết để debug (binary classification)
                    prob_summary = ", ".join([
                        f"{k}={v:.3f}" for k, v in probabilities.items() 
                        if k in ['benign', 'attack']
                    ])
                    logger.info(
                        f"[Sample {self.prediction_count:2d}] Final: {label} "
                        f"(encoded={encoded}, confidence={confidence:.3f}, probs=[{prob_summary}])"
                    )

                    self.send_prediction_result(prediction_result, original_key)
                    
                    if self.prediction_count % 10 == 0:
                        self._log_summary()

                except Exception as e:
                    logger.error(f"Error processing prediction: {e}")
                    continue

        except KeyboardInterrupt:
            logger.info("Level 1 prediction service stopped by user")
        except Exception as e:
            logger.error(f"Error in prediction loop: {e}")
        finally:
            self.stop()

    def _update_summary(self, prediction_result: dict):
        """Cập nhật summary statistics"""
        self.prediction_count += 1
        
        prediction = prediction_result.get('prediction', {})
        label = prediction.get('label', 'error')
        confidence = prediction.get('confidence', 0.0)
        
        if label in self.prediction_summary:
            self.prediction_summary[label] += 1
            if label != 'error' and confidence > 0:
                self.confidence_summary[label].append(confidence)
    
    def _log_summary(self):
        """Log summary statistics"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("LEVEL 1 PREDICTION SUMMARY (Random Forest):")
        logger.info("=" * 60)
        logger.info(f"Total predictions: {self.prediction_count}")
        logger.info("")
        logger.info("Prediction distribution:")
        for label, count in sorted(self.prediction_summary.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                percentage = (count / self.prediction_count) * 100 if self.prediction_count > 0 else 0
                avg_conf = 0.0
                if label in self.confidence_summary and len(self.confidence_summary[label]) > 0:
                    avg_conf = sum(self.confidence_summary[label]) / len(self.confidence_summary[label])
                logger.info(f"  - {label}: {count} ({percentage:.1f}%) - Avg confidence: {avg_conf:.3f}")
        logger.info("=" * 60)
        logger.info("")
    
    def stop(self):
        """Dừng service"""
        self.is_running = False
        
        if self.prediction_count > 0:
            logger.info("")
            self._log_summary()
        
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()
            logger.info("Level 1 prediction service (RF) stopped")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Safenet IDS - Level 1 Prediction Service (Random Forest)')
    parser.add_argument('--kafka-servers', default='localhost:9092',
                       help='Kafka bootstrap servers')
    parser.add_argument('--input-topic', default='preprocess_data',
                       help='Input topic name')
    parser.add_argument('--output-topic', default='level_1_predictions',
                       help='Output topic name')
    parser.add_argument('--model-path', default='artifacts_rf/ids_pipeline_rf.joblib',
                       help='Path to Level 1 Random Forest model file')
    parser.add_argument('--metadata-path', default='artifacts_rf/metadata.json',
                       help='Path to model metadata file')

    args = parser.parse_args()

    # Tạo thư mục logs nếu chưa có
    import os
    os.makedirs('services/logs', exist_ok=True)

    # Khởi tạo và chạy service
    service = Level1PredictionServiceRF(
        kafka_bootstrap_servers=args.kafka_servers,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        model_path=args.model_path,
        metadata_path=args.metadata_path
    )

    try:
        logger.info("Starting Safenet IDS Level 1 Prediction Service (Random Forest)...")
        service.start_prediction()
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service failed: {e}")


if __name__ == '__main__':
    main()

