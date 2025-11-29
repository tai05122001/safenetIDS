#!/usr/bin/env python3
"""
Safenet IDS - Alerting Service
Đọc từ topic level2_predictions, tạo alerts và gửi đến topic ids_alerts
"""

import json
import logging
import sqlite3
import os
from pathlib import Path
from kafka import KafkaConsumer, KafkaProducer
from datetime import datetime, timedelta
from typing import Dict, Any, List
from collections import defaultdict

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('services/logs/alerting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Alerting')

class AlertingService:
    """Service tạo và quản lý alerts từ kết quả predictions"""

    def __init__(self,
                 kafka_bootstrap_servers='127.0.0.1:9092',
                 input_topic='level_2_predictions',
                 output_topic='alert',
                 db_path='services/data/alerts.db',
                 alert_thresholds=None):
        """
        Khởi tạo Alerting Service

        Args:
            kafka_bootstrap_servers: Kafka bootstrap servers
            input_topic: Topic để đọc kết quả Level 2 predictions
            output_topic: Topic để gửi alerts
            db_path: Đường dẫn database để lưu alerts
            alert_thresholds: Ngưỡng confidence để tạo alert
        """
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.db_path = Path(db_path)
        self.consumer = None
        self.producer = None
        self.kafka_servers = kafka_bootstrap_servers
        self.is_running = False

        # Database connection
        self.db_conn = None

        # Alert thresholds (confidence > threshold để tạo alert)
        self.alert_thresholds = alert_thresholds or {
            'benign': 0.0,  # Không tạo alert cho benign
            'dos': 0.7,
            'ddos': 0.6,
            'bot': 0.75,
            'rare_attack': 0.8,
            'default': 0.7
        }

        # Thống kê alerts
        self.alert_stats = defaultdict(int)
        self.recent_alerts = []

        # Khởi tạo database và Kafka clients
        self._init_database()
        self._init_consumer()
        self._init_producer()

    def _init_database(self):
        """Khởi tạo database để lưu alerts"""
        try:
            # Tạo thư mục nếu chưa có
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            self.db_conn = sqlite3.connect(str(self.db_path))
            cursor = self.db_conn.cursor()

            # Tạo bảng alerts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE,
                    timestamp TEXT,
                    severity TEXT,
                    attack_type TEXT,
                    confidence REAL,
                    source_ip TEXT,
                    destination_ip TEXT,
                    source_port INTEGER,
                    destination_port INTEGER,
                    protocol TEXT,
                    description TEXT,
                    raw_data TEXT,
                    status TEXT DEFAULT 'active'
                )
            ''')

            # Tạo bảng alert_stats
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    attack_type TEXT,
                    count INTEGER,
                    time_window TEXT
                )
            ''')

            self.db_conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def _init_consumer(self):
        """Khởi tạo Kafka consumer"""
        try:
            self.consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.kafka_servers,
                group_id='safenet-ids-alerting-group',
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

    def _calculate_severity(self, attack_type: str, confidence: float) -> str:
        """
        Tính toán mức độ severity của alert

        Args:
            attack_type: Loại attack
            confidence: Độ tin cậy của prediction

        Returns:
            Mức độ severity: 'low', 'medium', 'high', 'critical'
        """
        # Base severity theo loại attack
        base_severity = {
            'benign': 'none',
            'dos': 'high',
            'ddos': 'critical',
            'bot': 'medium',
            'rare_attack': 'high'
        }

        severity = base_severity.get(attack_type, 'medium')

        # Điều chỉnh theo confidence
        if confidence > 0.9:
            if severity == 'high':
                severity = 'critical'
        elif confidence < 0.6:
            if severity == 'high':
                severity = 'medium'
            elif severity == 'medium':
                severity = 'low'

        return severity

    def _should_create_alert(self, prediction: Dict[str, Any]) -> bool:
        """
        Kiểm tra xem có nên tạo alert không

        Args:
            prediction: Kết quả prediction

        Returns:
            True nếu nên tạo alert
        """
        attack_type = prediction.get('label', '').lower()

        # Không tạo alert cho benign
        if 'benign' in attack_type or attack_type == 'error':
            return False

        confidence = prediction.get('confidence', 0.0)
        threshold = self.alert_thresholds.get(attack_type, self.alert_thresholds['default'])

        return confidence >= threshold

    def _extract_network_info(self, level2_result: Dict[str, Any]) -> Dict[str, str]:
        """
        Trích xuất thông tin network từ kết quả prediction

        Args:
            level2_result: Kết quả Level 2 prediction

        Returns:
            Dictionary chứa thông tin network
        """
        # Lấy từ level1_result -> original_data
        level1_result = level2_result.get('level1_result', {})
        original_data = level1_result.get('original_data', {})

        return {
            'source_ip': original_data.get('source_ip', 'unknown'),
            'destination_ip': original_data.get('destination_ip', 'unknown'),
            'source_port': original_data.get('source_port', 0),
            'destination_port': original_data.get('destination_port', 0),
            'protocol': original_data.get('protocol', 'unknown')
        }

    def _generate_alert_description(self, level2_result: Dict[str, Any]) -> str:
        """
        Tạo mô tả cho alert

        Args:
            level2_result: Kết quả Level 2 prediction

        Returns:
            Mô tả alert
        """
        level2_pred = level2_result.get('level2_prediction', {})
        level1_pred = level2_result.get('level1_result', {}).get('prediction', {})

        attack_type = level2_pred.get('label', 'Unknown')
        confidence = level2_pred.get('confidence', 0.0)
        group = level2_pred.get('group', 'unknown')

        network_info = self._extract_network_info(level2_result)

        description = (
            f"IDS Alert: {attack_type} detected. "
            f"Attack group: {group}. "
            f"Confidence: {confidence:.2f}. "
            f"Source: {network_info['source_ip']}:{network_info['source_port']} -> "
            f"Destination: {network_info['destination_ip']}:{network_info['destination_port']} "
            f"(Protocol: {network_info['protocol']})"
        )

        return description

    def _create_alert(self, level2_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tạo structured alert từ kết quả prediction Level 2

        Process chi tiết:
        1. Data Extraction:
           - Parse level2_result để lấy prediction details
           - Extract network information (IP, port, protocol)
           - Get attack type và confidence từ model output

        2. Severity Calculation:
           - Map attack type sang base severity level
           - Adjust theo confidence score
           - Apply business logic rules

        3. Alert Structuring:
           - Generate unique alert ID
           - Create descriptive alert message
           - Include all relevant metadata
           - Set status và priority flags

        Args:
            level2_result (Dict[str, Any]): Kết quả prediction từ Level 2 service

        Returns:
            Dict[str, Any]: Complete alert object với format:
            {
                'alert_id': unique identifier,
                'timestamp': ISO timestamp,
                'severity': 'low'|'medium'|'high'|'critical',
                'attack_type': specific attack name,
                'confidence': prediction confidence,
                'source_ip': attacker IP,
                'destination_ip': target IP,
                'description': human readable alert text,
                'status': 'active',
                ...
            }
        """
        timestamp = datetime.now().isoformat()
        level2_pred = level2_result.get('level2_prediction', {})

        # Tạo alert ID unique
        alert_id = f"IDS-{int(datetime.now().timestamp() * 1000)}"

        # Thông tin prediction
        attack_type = level2_pred.get('label', 'Unknown')
        confidence = level2_pred.get('confidence', 0.0)

        # Tính severity
        severity = self._calculate_severity(attack_type.lower(), confidence)

        # Thông tin network
        network_info = self._extract_network_info(level2_result)

        # Mô tả alert
        description = self._generate_alert_description(level2_result)

        alert = {
            'alert_id': alert_id,
            'timestamp': timestamp,
            'severity': severity,
            'attack_type': attack_type,
            'confidence': confidence,
            'source_ip': network_info['source_ip'],
            'destination_ip': network_info['destination_ip'],
            'source_port': network_info['source_port'],
            'destination_port': network_info['destination_port'],
            'protocol': network_info['protocol'],
            'description': description,
            'raw_prediction_data': level2_result,
            'status': 'active'
        }

        return alert

    def _save_alert_to_db(self, alert: Dict[str, Any]):
        """
        Lưu alert vào database

        Args:
            alert: Thông tin alert
        """
        try:
            cursor = self.db_conn.cursor()

            cursor.execute('''
                INSERT INTO alerts (
                    alert_id, timestamp, severity, attack_type, confidence,
                    source_ip, destination_ip, source_port, destination_port,
                    protocol, description, raw_data, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert['alert_id'],
                alert['timestamp'],
                alert['severity'],
                alert['attack_type'],
                alert['confidence'],
                alert['source_ip'],
                alert['destination_ip'],
                alert['source_port'],
                alert['destination_port'],
                alert['protocol'],
                alert['description'],
                json.dumps(alert['raw_prediction_data']),
                alert['status']
            ))

            self.db_conn.commit()

            # Cập nhật thống kê
            attack_type = alert['attack_type']
            self.alert_stats[attack_type] += 1

            logger.info(f"Alert saved to database: {alert['alert_id']}")

        except Exception as e:
            logger.error(f"Failed to save alert to database: {e}")

    def _update_alert_stats(self):
        """Cập nhật thống kê alerts vào database"""
        try:
            current_time = datetime.now()
            time_window = current_time.strftime('%Y-%m-%d %H:00:00')

            cursor = self.db_conn.cursor()

            for attack_type, count in self.alert_stats.items():
                cursor.execute('''
                    INSERT INTO alert_stats (timestamp, attack_type, count, time_window)
                    VALUES (?, ?, ?, ?)
                ''', (current_time.isoformat(), attack_type, count, time_window))

            self.db_conn.commit()

            # Reset stats sau khi lưu
            self.alert_stats.clear()

        except Exception as e:
            logger.error(f"Failed to update alert stats: {e}")

    def send_alert(self, alert: Dict[str, Any], original_key: str = None):
        """
        Gửi alert đến Kafka topic

        Args:
            alert: Thông tin alert
            original_key: Key từ message gốc
        """
        try:
            key = original_key or alert.get('alert_id', str(datetime.now().timestamp()))

            future = self.producer.send(self.output_topic, value=alert, key=key)
            record_metadata = future.get(timeout=10)

            logger.info(f"Sent alert {alert['alert_id']} to {record_metadata.topic} "
                       f"partition {record_metadata.partition} "
                       f"offset {record_metadata.offset}")

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def process_level2_result(self, level2_result: Dict[str, Any], original_key: str = None):
        """
        Xử lý kết quả từ Level 2 prediction và tạo alert

        Args:
            level2_result: Kết quả Level 2 prediction
            original_key: Key của message gốc
        """
        try:
            # Kiểm tra có phải kết quả Level 2 hay không
            if 'level2_prediction' in level2_result:
                # Đây là kết quả Level 2 - kiểm tra có nên tạo alert không
                level2_pred = level2_result['level2_prediction']

                if self._should_create_alert(level2_pred):
                    # Tạo alert
                    alert = self._create_alert(level2_result)

                    # Lưu vào database
                    self._save_alert_to_db(alert)

                    # Gửi đến Kafka
                    self.send_alert(alert, original_key)

                    logger.info(f"Alert created for attack: {alert['attack_type']} "
                              f"(severity: {alert['severity']}, confidence: {alert['confidence']:.2f})")
                else:
                    confidence = level2_pred.get('confidence', 0.0)
                    logger.info(f"Alert not created - confidence too low: {confidence:.2f}")
            else:
                # Có thể là kết quả Level 1 mà không có Level 2 (benign, ddos, bot)
                level1_result = level2_result
                level1_pred = level1_result.get('prediction', {})

                if self._should_create_alert(level1_pred):
                    # Tạo alert từ Level 1 result
                    alert = self._create_alert_from_level1(level1_result)
                    self._save_alert_to_db(alert)
                    self.send_alert(alert, original_key)

                    logger.info(f"Alert created from Level 1 for attack: {alert['attack_type']}")
                else:
                    logger.debug("No alert needed for this prediction")

        except Exception as e:
            logger.error(f"Error processing Level 2 result: {e}")

    def _create_alert_from_level1(self, level1_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tạo alert từ kết quả Level 1 (cho các attack không cần Level 2)

        Args:
            level1_result: Kết quả Level 1 prediction

        Returns:
            Dictionary chứa thông tin alert
        """
        timestamp = datetime.now().isoformat()
        level1_pred = level1_result.get('prediction', {})
        original_data = level1_result.get('original_data', {})

        alert_id = f"IDS-L1-{int(datetime.now().timestamp() * 1000)}"

        attack_type = level1_pred.get('label', 'Unknown')
        confidence = level1_pred.get('confidence', 0.0)
        severity = self._calculate_severity(attack_type.lower(), confidence)

        alert = {
            'alert_id': alert_id,
            'timestamp': timestamp,
            'severity': severity,
            'attack_type': attack_type,
            'confidence': confidence,
            'source_ip': original_data.get('source_ip', 'unknown'),
            'destination_ip': original_data.get('destination_ip', 'unknown'),
            'source_port': original_data.get('source_port', 0),
            'destination_port': original_data.get('destination_port', 0),
            'protocol': original_data.get('protocol', 'unknown'),
            'description': f"IDS Alert (Level 1): {attack_type} detected with confidence {confidence:.2f}",
            'raw_prediction_data': level1_result,
            'status': 'active'
        }

        return alert

    def start_alerting(self):
        """Bắt đầu quá trình alerting"""
        self.is_running = True
        logger.info(f"Starting alerting service: {self.input_topic} -> {self.output_topic}")

        stats_update_interval = 300  # 5 minutes
        last_stats_update = datetime.now()

        try:
            for message in self.consumer:
                if not self.is_running:
                    break

                try:
                    # Lấy dữ liệu từ message
                    level2_result = message.value
                    original_key = message.key

                    logger.debug(f"Processing alert for key: {original_key}")

                    # Xử lý và tạo alert
                    self.process_level2_result(level2_result, original_key)

                    # Cập nhật stats định kỳ
                    if (datetime.now() - last_stats_update).seconds >= stats_update_interval:
                        self._update_alert_stats()
                        last_stats_update = datetime.now()

                except Exception as e:
                    logger.error(f"Error processing alert: {e}")
                    continue

        except KeyboardInterrupt:
            logger.info("Alerting service stopped by user")
        except Exception as e:
            logger.error(f"Error in alerting loop: {e}")
        finally:
            # Cập nhật stats lần cuối
            self._update_alert_stats()
            self.stop()

    def stop(self):
        """Dừng service"""
        self.is_running = False
        if self.db_conn:
            self.db_conn.close()
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()
            logger.info("Alerting service stopped")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Safenet IDS - Alerting Service')
    parser.add_argument('--kafka-servers', default='127.0.0.1:9092',
                       help='Kafka bootstrap servers')
    parser.add_argument('--input-topic', default='level_2_predictions',
                       help='Input topic name (Level 2 predictions)')
    parser.add_argument('--output-topic', default='alert',
                       help='Output topic name (IDS alerts)')
    parser.add_argument('--db-path', default='services/data/alerts.db',
                       help='Path to alerts database')
    parser.add_argument('--alert-threshold', type=float, default=0.7,
                       help='Default confidence threshold for alerts')

    args = parser.parse_args()

    # Tạo thư mục logs và data nếu chưa có
    import os
    os.makedirs('services/logs', exist_ok=True)
    os.makedirs('services/data', exist_ok=True)

    # Cập nhật thresholds
    thresholds = {
        'benign': 0.0,
        'dos': 0.7,
        'ddos': 0.6,
        'bot': 0.75,
        'rare_attack': 0.8,
        'default': args.alert_threshold
    }

    # Khởi tạo và chạy service
    service = AlertingService(
        kafka_bootstrap_servers=args.kafka_servers,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        db_path=args.db_path,
        alert_thresholds=thresholds
    )

    try:
        logger.info("Starting Safenet IDS Alerting Service...")
        service.start_alerting()
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service failed: {e}")


if __name__ == '__main__':
    main()
