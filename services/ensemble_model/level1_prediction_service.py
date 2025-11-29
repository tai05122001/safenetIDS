#!/usr/bin/env python3
"""
Safenet IDS - Level 1 Prediction Service
Dịch vụ chạy prediction Level 1 (phân loại nhóm attack tổng quát)

Chức năng chính:
- Đọc dữ liệu đã tiền xử lý từ Kafka topic 'preprocessed_events'
- Load và chạy model RandomForest Level 1 từ artifacts
- Phân loại thành 4 nhóm: benign, dos, ddos, portscan
- Gửi kết quả prediction kèm confidence scores đến 'level1_predictions'

Model Level 1:
- Algorithm: RandomForestClassifier (ensemble learning)
- Input: 78+ network flow features đã được preprocessing
- Output: Encoded prediction (0-3) + confidence probabilities
- Training data: CICIDS2017 với hierarchical labels (đã bỏ Bot, Infiltration, Heartbleed)

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
from collections import Counter  # Count votes for hard voting
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
            3: 'portscan'
        }
        
        # Summary statistics
        self.prediction_count = 0
        self.prediction_summary = {
            'benign': 0,
            'dos': 0,
            'ddos': 0,
            'portscan': 0,
            'error': 0
        }
        self.confidence_summary = {
            'benign': [],
            'dos': [],
            'ddos': [],
            'portscan': []
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
            # Lấy feature names từ model pipeline (ColumnTransformer)
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
                    
                    # Tạo dummy data với tất cả features cần thiết
                    dummy_data_dict = {col: 0.0 for col in feature_names}
                    # Thêm một số giá trị mặc định cho các features quan trọng
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
            # Phải khớp với drop_columns_resolved trong metadata.json
            exclude_cols = [
                'timestamp', 'source_ip', 'destination_ip',  # Identifiers
                'label', 'label_group',  # Target variables (theo metadata.json)
                'label_encoded', 'label_group_encoded',  # Encoded labels (không dùng để predict)
                'preprocessing_timestamp', 'preprocessing_metadata', 'preprocessing_error'  # Metadata
            ]
            
            # Thêm các cột từ metadata nếu có
            if self.metadata and 'drop_columns_resolved' in self.metadata:
                metadata_drop_cols = self.metadata['drop_columns_resolved']
                for col in metadata_drop_cols:
                    if col not in exclude_cols:
                        exclude_cols.append(col)
                        logger.debug(f"Added drop column from metadata: {col}")

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
            
            # Debug: Log số lượng features và một số giá trị mẫu
            logger.info(f"Using {len(feature_cols)} features for prediction")
            if len(feature_cols) < 50:
                logger.warning(f"Only {len(feature_cols)} features found - model may need more features!")
            
            # Log một số features mẫu để debug
            logger.debug(f"Sample features: {feature_cols[:10]}")
            logger.debug(f"Excluded columns: {exclude_cols}")

            # ===== PHASE 2: Model Inference =====
            # Log thông tin về features trước khi predict
            logger.info(f"Predicting with {len(feature_cols)} features")
            logger.debug(f"Feature names (first 20): {feature_cols[:20]}")
            
            # Kiểm tra nếu là VotingClassifier - log từng estimator
            classifier = self.model.named_steps.get('classifier', None)
            if classifier is not None and hasattr(classifier, 'estimators_'):
                logger.info("Detected VotingClassifier - logging individual estimator predictions")
                
                # Transform features qua preprocessor
                # LƯU Ý: Preprocessor có thể yêu cầu các cột đã được fit, kể cả label_encoded
                # Nên cần transform qua toàn bộ pipeline, không transform riêng lẻ
                preprocessor = self.model.named_steps.get('preprocess', None)
                if preprocessor is not None:
                    try:
                        # Thử transform với features_df hiện tại
                        X_transformed = preprocessor.transform(features_df)
                    except (ValueError, KeyError) as e:
                        # Nếu lỗi thiếu cột, thử với DataFrame đầy đủ hơn (bao gồm label_encoded nếu có)
                        logger.debug(f"Transform failed with features_df, trying with full df: {e}")
                        # Tạo DataFrame với tất cả cột numeric (kể cả label_encoded) để transform
                        # Nhưng chỉ dùng các cột features để predict
                        all_numeric_cols = [col for col in df.columns 
                                          if col not in ['preprocessing_timestamp', 'preprocessing_metadata', 'preprocessing_error']
                                          and df[col].dtype in ['int64', 'float64', 'float32', 'int32']]
                        transform_df = df[all_numeric_cols]
                        X_transformed = preprocessor.transform(transform_df)
                        logger.debug(f"Successfully transformed with {len(all_numeric_cols)} columns")
                else:
                    X_transformed = features_df.values if isinstance(features_df, pd.DataFrame) else features_df
                
                # Đảm bảo X_transformed là numpy array 2D
                if isinstance(X_transformed, pd.DataFrame):
                    X_transformed = X_transformed.values
                if hasattr(X_transformed, 'ndim') and X_transformed.ndim == 1:
                    X_transformed = X_transformed.reshape(1, -1)
                if not isinstance(X_transformed, np.ndarray):
                    X_transformed = np.array(X_transformed)
                
                logger.info(f"Transformed features shape: {X_transformed.shape}")
                
                # Lấy predictions từ từng estimator
                estimator_predictions = {}
                estimator_probas = {}
                
                # Thử dùng named_estimators_ trước
                if hasattr(classifier, 'named_estimators_'):
                    estimators_dict = classifier.named_estimators_
                    for name, estimator in estimators_dict.items():
                        try:
                            pred = estimator.predict(X_transformed)[0]
                            proba = estimator.predict_proba(X_transformed)[0]
                            pred_label = self.label_mapping.get(int(pred), f'unknown_{pred}')
                            estimator_predictions[name] = {
                                'encoded': int(pred),
                                'label': pred_label,
                                'confidence': float(max(proba))
                            }
                            estimator_probas[name] = proba
                            logger.info(f"  {name}: predicts {pred_label} (encoded={pred}, confidence={max(proba):.3f})")
                        except Exception as e:
                            logger.warning(f"  {name}: Error getting prediction - {e}")
                            continue
                elif hasattr(classifier, 'estimators_'):
                    estimators_list = classifier.estimators_
                    for idx, item in enumerate(estimators_list):
                        try:
                            estimator = None
                            name = f"estimator_{idx}"
                            
                            if isinstance(item, tuple):
                                if len(item) >= 2:
                                    name = str(item[0])
                                    estimator = item[-1]
                                elif len(item) == 1:
                                    estimator = item[0]
                            else:
                                estimator = item
                            
                            if estimator is None:
                                continue
                            
                            pred = estimator.predict(X_transformed)[0]
                            proba = estimator.predict_proba(X_transformed)[0]
                            pred_label = self.label_mapping.get(int(pred), f'unknown_{pred}')
                            estimator_predictions[name] = {
                                'encoded': int(pred),
                                'label': pred_label,
                                'confidence': float(max(proba))
                            }
                            estimator_probas[name] = proba
                            logger.info(f"  {name}: predicts {pred_label} (encoded={pred}, confidence={max(proba):.3f})")
                        except Exception as e:
                            logger.warning(f"  estimator_{idx}: Error - {e}")
                            continue
                
                # Đếm votes
                votes = Counter([pred['label'] for pred in estimator_predictions.values()])
                logger.info(f"Votes: {dict(votes)}")
                logger.info(f"Final prediction will be: {votes.most_common(1)[0][0] if votes else 'unknown'}")
            
            # Chạy prediction - trả về encoded class (0-4)
            # LƯU Ý: model.predict() cần DataFrame có đủ cột mà preprocessor yêu cầu
            # Phải dùng cùng DataFrame với phần logging để đảm bảo nhất quán
            # Nếu đã transform ở trên (cho logging), dùng cùng DataFrame đó
            if classifier is not None and hasattr(classifier, 'estimators_'):
                # Đã transform ở trên, dùng cùng DataFrame
                # Tạo DataFrame với tất cả cột numeric (kể cả label_encoded) để predict
                all_numeric_cols = [col for col in df.columns 
                                  if col not in ['preprocessing_timestamp', 'preprocessing_metadata', 'preprocessing_error']
                                  and df[col].dtype in ['int64', 'float64', 'float32', 'int32']]
                predict_df = df[all_numeric_cols]
                logger.debug(f"Using {len(all_numeric_cols)} columns for model.predict() (same as logging)")
            else:
                # Không phải VotingClassifier, dùng features_df
                predict_df = features_df
            
            try:
                prediction_encoded = self.model.predict(predict_df)[0]
            except (ValueError, KeyError) as e:
                if 'columns are missing' in str(e) or 'label_encoded' in str(e):
                    logger.debug(f"model.predict() failed, trying with full df: {e}")
                    # Tạo DataFrame với tất cả cột numeric (kể cả label_encoded) để predict
                    all_numeric_cols = [col for col in df.columns 
                                      if col not in ['preprocessing_timestamp', 'preprocessing_metadata', 'preprocessing_error']
                                      and df[col].dtype in ['int64', 'float64', 'float32', 'int32']]
                    predict_df = df[all_numeric_cols]
                    prediction_encoded = self.model.predict(predict_df)[0]
                    logger.debug(f"Successfully predicted with {len(all_numeric_cols)} columns")
                else:
                    raise
            
            # Debug: Log prediction encoded
            logger.info(f"Final prediction: {prediction_encoded} -> {self.label_mapping.get(int(prediction_encoded), 'unknown')}")

            # Chạy predict probabilities - confidence scores cho tất cả classes
            # Lưu ý: VotingClassifier với voting="hard" không có predict_proba
            # Cần tính từ các base estimators hoặc dùng voting="soft"
            try:
                prediction_proba = self.model.predict_proba(features_df)[0]
            except (AttributeError, TypeError) as e:
                # Model không có predict_proba (hard voting) - tính từ base estimators
                logger.debug(f"Model không hỗ trợ predict_proba ({type(e).__name__}), tính từ base estimators")
                classifier = self.model.named_steps.get('classifier', None)
                
                if classifier is not None and hasattr(classifier, 'estimators_'):
                    # VotingClassifier - tính trung bình probabilities từ các base models
                    # Transform features qua preprocessor một lần
                    preprocessor = self.model.named_steps.get('preprocess', None)
                    if preprocessor is not None:
                        try:
                            # Thử transform với features_df hiện tại
                            X_transformed = preprocessor.transform(features_df)
                        except (ValueError, KeyError) as e:
                            # Nếu lỗi thiếu cột, thử với DataFrame đầy đủ hơn
                            logger.debug(f"Transform failed with features_df, trying with full df: {e}")
                            # Tạo DataFrame với tất cả cột numeric (kể cả label_encoded) để transform
                            all_numeric_cols = [col for col in df.columns 
                                              if col not in ['preprocessing_timestamp', 'preprocessing_metadata', 'preprocessing_error']
                                              and df[col].dtype in ['int64', 'float64', 'float32', 'int32']]
                            transform_df = df[all_numeric_cols]
                            X_transformed = preprocessor.transform(transform_df)
                            logger.debug(f"Successfully transformed with {len(all_numeric_cols)} columns")
                    else:
                        X_transformed = features_df.values if isinstance(features_df, pd.DataFrame) else features_df
                    
                    # Đảm bảo X_transformed là numpy array 2D (không phải DataFrame)
                    if isinstance(X_transformed, pd.DataFrame):
                        X_transformed = X_transformed.values
                    if hasattr(X_transformed, 'ndim') and X_transformed.ndim == 1:
                        X_transformed = X_transformed.reshape(1, -1)
                    # Đảm bảo là numpy array
                    if not isinstance(X_transformed, np.ndarray):
                        X_transformed = np.array(X_transformed)
                    
                    all_probas = []
                    # VotingClassifier có thể có estimators_ hoặc named_estimators_
                    # Thử dùng named_estimators_ trước (an toàn hơn, có tên)
                    if hasattr(classifier, 'named_estimators_'):
                        # named_estimators_ là dict {name: estimator}
                        estimators_dict = classifier.named_estimators_
                        for name, estimator in estimators_dict.items():
                            try:
                                proba = estimator.predict_proba(X_transformed)[0]
                                all_probas.append(proba)
                                # Log chi tiết probabilities từ từng estimator
                                max_prob_idx = np.argmax(proba)
                                max_prob_label = self.label_mapping.get(max_prob_idx, f'unknown_{max_prob_idx}')
                                logger.debug(f"  {name} proba: {max_prob_label}={proba[max_prob_idx]:.3f}, full={[f'{p:.3f}' for p in proba]}")
                            except Exception as est_e:
                                logger.warning(f"Could not get proba from {name}: {est_e}")
                                continue
                    elif hasattr(classifier, 'estimators_'):
                        # estimators_ có thể là list các tuple (name, estimator) hoặc list các estimator
                        estimators_list = classifier.estimators_
                        
                        if estimators_list and len(estimators_list) > 0:
                            # Kiểm tra cấu trúc phần tử đầu tiên
                            first_item = estimators_list[0]
                            
                            # Xử lý từng item một cách an toàn
                            for idx, item in enumerate(estimators_list):
                                try:
                                    estimator = None
                                    name = f"estimator_{idx}"
                                    
                                    # Kiểm tra và extract estimator
                                    if isinstance(item, tuple):
                                        # Tuple - lấy phần tử cuối (estimator)
                                        if len(item) >= 2:
                                            name = str(item[0]) if len(item) > 0 else name
                                            estimator = item[-1]
                                        elif len(item) == 1:
                                            estimator = item[0]
                                        else:
                                            logger.warning(f"Empty tuple at index {idx}, skipping")
                                            continue
                                    else:
                                        # Không phải tuple - dùng trực tiếp
                                        estimator = item
                                    
                                    if estimator is None:
                                        logger.warning(f"Could not extract estimator at index {idx}, skipping")
                                        continue
                                    
                                    # Lấy probabilities từ estimator
                                    proba = estimator.predict_proba(X_transformed)[0]
                                    all_probas.append(proba)
                                    # Log chi tiết probabilities từ từng estimator
                                    max_prob_idx = np.argmax(proba)
                                    max_prob_label = self.label_mapping.get(max_prob_idx, f'unknown_{max_prob_idx}')
                                    logger.debug(f"  {name} proba: {max_prob_label}={proba[max_prob_idx]:.3f}, full={[f'{p:.3f}' for p in proba]}")
                                    
                                except Exception as est_e:
                                    logger.warning(f"Could not get proba from estimator {idx}: {est_e}")
                                    continue
                    
                    if all_probas:
                        # Trung bình probabilities từ tất cả base models
                        prediction_proba = np.mean(all_probas, axis=0)
                        logger.info(f"Computed averaged probabilities from {len(all_probas)} base estimators")
                        # Log probabilities cuối cùng
                        for idx, prob in enumerate(prediction_proba):
                            label = self.label_mapping.get(idx, f'unknown_{idx}')
                            logger.debug(f"  Final proba: {label}={prob:.3f}")
                    else:
                        # Fallback: tạo uniform probabilities
                        num_classes = len(self.label_mapping)
                        prediction_proba = np.ones(num_classes) / num_classes
                        logger.warning("Could not compute probabilities, using uniform distribution")
                else:
                    # Fallback: tạo uniform probabilities
                    num_classes = len(self.label_mapping)
                    prediction_proba = np.ones(num_classes) / num_classes
                    logger.warning("Could not compute probabilities from model, using uniform distribution")

            # Map encoded prediction sang readable label
            prediction_label = self.label_mapping.get(int(prediction_encoded), 'unknown')
            
            # Debug: Log full probability distribution để kiểm tra
            logger.debug(f"Full probabilities: {dict(zip([self.label_mapping[i] for i in range(len(prediction_proba))], prediction_proba))}")

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

                    # Cập nhật summary statistics
                    self._update_summary(prediction_result)

                    # Log mỗi prediction thành công với thông tin chi tiết
                    prediction = prediction_result.get('prediction', {})
                    label = prediction.get('label', 'unknown')
                    confidence = prediction.get('confidence', 0.0)
                    encoded = prediction.get('encoded', -1)
                    logger.info(f"[Sample {self.prediction_count:2d}] Final: {label} (encoded={encoded}, confidence={confidence:.3f})")

                    # Gửi kết quả
                    self.send_prediction_result(prediction_result, original_key)
                    
                    # Log summary mỗi 10 predictions
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
        logger.info("LEVEL 1 PREDICTION SUMMARY:")
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
        
        # Log final summary
        if self.prediction_count > 0:
            logger.info("")
            self._log_summary()
        
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
