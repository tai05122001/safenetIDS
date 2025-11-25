#!/usr/bin/env python3
"""
Safenet IDS - Network Data Producer Service
Dịch vụ thu thập dữ liệu mạng và gửi đến Kafka topic 'raw_network_events'

Chức năng chính:
- Tạo dữ liệu mạng mẫu (hoặc đọc từ file) mô phỏng traffic mạng
- Gửi dữ liệu đến Kafka để các service khác xử lý
- Hỗ trợ cấu hình linh hoạt cho môi trường testing/production
"""

# Import các thư viện cần thiết
import json  # Xử lý dữ liệu JSON cho Kafka messages
import time  # Quản lý thời gian và delays
import logging  # Ghi log hoạt động của service
import random  # Tạo dữ liệu ngẫu nhiên cho testing
from datetime import datetime  # Xử lý timestamps
from kafka import KafkaProducer  # Kafka producer để gửi dữ liệu
import pandas as pd  # Xử lý dữ liệu dạng bảng (cho historical data)
import numpy as np  # Hỗ trợ tính toán số học

# Cấu hình logging để theo dõi hoạt động của service
# - Level INFO: Ghi các thông tin quan trọng như kết nối, gửi dữ liệu thành công
# - Format: Bao gồm timestamp, tên logger, level, và message
# - Handlers: Ghi ra file và console để dễ debug
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('services/logs/network_producer.log'),  # Log file
        logging.StreamHandler()  # Console output
    ]
)

# Tạo logger instance cho service này
logger = logging.getLogger('NetworkProducer')

class NetworkDataProducer:
    """
    Lớp chính của Network Data Producer Service

    Trách nhiệm:
    - Quản lý kết nối Kafka
    - Tạo hoặc đọc dữ liệu network
    - Gửi dữ liệu đến Kafka topic
    - Xử lý lỗi và logging
    """

    def __init__(self, kafka_bootstrap_servers='localhost:9092', topic='raw_network_events'):
        """
        Khởi tạo Network Data Producer instance

        Args:
            kafka_bootstrap_servers (str): Địa chỉ Kafka servers, mặc định localhost:9092
            topic (str): Tên Kafka topic để gửi dữ liệu, mặc định 'raw_network_events'
        """
        # Lưu cấu hình cơ bản
        self.topic = topic  # Tên topic đích
        self.producer = None  # Kafka producer instance (sẽ được khởi tạo sau)
        self.kafka_servers = kafka_bootstrap_servers  # Địa chỉ Kafka servers
        self.is_running = False  # Cờ kiểm tra service đang chạy hay không

        # Khởi tạo Kafka producer ngay khi tạo instance
        self._init_producer()

    def _init_producer(self):
        """
        Khởi tạo Kafka producer với cấu hình tối ưu

        Cấu hình chi tiết:
        - bootstrap_servers: Địa chỉ Kafka cluster
        - value_serializer: Chuyển đổi Python dict thành JSON bytes cho message value
        - key_serializer: Chuyển đổi key thành string bytes
        - acks='all': Đảm bảo message được ghi vào tất cả replicas (độ tin cậy cao)
        - retries=3: Thử lại tối đa 3 lần khi gặp lỗi
        - linger_ms=5: Chờ 5ms để batch messages trước khi gửi
        - batch_size=32768: Kích thước batch tối đa 32KB
        - buffer_memory=67108864: Buffer 64MB cho messages chờ gửi
        """
        try:
            # Tạo Kafka producer với cấu hình tối ưu cho throughput và reliability
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers,  # Kết nối đến Kafka cluster
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),  # Serialize dict -> JSON bytes
                key_serializer=lambda k: str(k).encode('utf-8'),  # Serialize key -> string bytes
                acks='all',  # Đảm bảo message được replicate đầy đủ
                retries=3,  # Retry khi gặp lỗi network/transient
                linger_ms=5,  # Batch delay để tăng throughput
                batch_size=32768,  # Kích thước batch tối ưu
                buffer_memory=67108864  # Memory buffer cho queuing
            )

            # Log thành công
            logger.info(f"Kafka producer initialized successfully for topic: {self.topic}")

        except Exception as e:
            # Log lỗi và re-raise để caller xử lý
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise

    def generate_sample_network_data(self):
        """
        Tạo mẫu dữ liệu network traffic để testing/demo

        Mô phỏng các đặc trưng của dataset CICIDS2017 bao gồm:
        - Thông tin kết nối cơ bản (IP, port, protocol)
        - Thống kê packet (số lượng, kích thước)
        - Timing information (duration, inter-arrival times)
        - TCP flags và header information

        Trong môi trường production, method này nên được thay thế bằng:
        - Đọc từ PCAP files
        - Capture trực tiếp từ network interface
        - Tích hợp với network monitoring tools

        Returns:
            dict: Dictionary chứa đầy đủ thông tin network flow
        """
        # Tạo timestamp hiện tại để đánh dấu thời điểm capture
        timestamp = datetime.now().isoformat()

        # Hàm helper tạo địa chỉ IP ngẫu nhiên
        # Format: xxx.xxx.xxx.xxx với các octet trong khoảng hợp lệ
        def random_ip():
            return f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}"

        # Tạo dữ liệu network flow sample dựa trên schema của CICIDS2017
        # Bao gồm tất cả các features quan trọng cho IDS classification
        network_data = {
            # Thông tin thời gian
            'timestamp': timestamp,  # Thời điểm capture flow

            # Thông tin kết nối cơ bản (5-tuple)
            'source_ip': random_ip(),  # IP nguồn (client)
            'destination_ip': random_ip(),  # IP đích (server)
            'source_port': random.randint(1024, 65535),  # Port nguồn (ephemeral ports)
            'destination_port': random.randint(1, 1024),  # Port đích (well-known ports)
            'protocol': random.choice([6, 17, 1]),  # Protocol: 6=TCP, 17=UDP, 1=ICMP
            # Thông tin flow characteristics (đặc trưng của flow)
            'flow_duration': random.randint(1, 1000000),  # Thời gian flow tồn tại (microseconds)

            # Thống kê packets
            'total_fwd_packets': random.randint(1, 100),  # Tổng packets từ source -> destination
            'total_backward_packets': random.randint(1, 100),  # Tổng packets từ destination -> source
            'total_length_of_fwd_packets': random.randint(0, 10000),  # Tổng bytes forward
            'total_length_of_bwd_packets': random.randint(0, 10000),  # Tổng bytes backward

            # Thống kê kích thước packets (forward direction)
            'fwd_packet_length_max': random.randint(0, 1500),  # Packet lớn nhất forward (MTU=1500)
            'fwd_packet_length_min': random.randint(0, 100),  # Packet nhỏ nhất forward
            'fwd_packet_length_mean': random.uniform(0, 1000),  # Trung bình kích thước forward
            'fwd_packet_length_std': random.uniform(0, 500),  # Độ lệch chuẩn kích thước forward
            # Thống kê kích thước packets (backward direction)
            'bwd_packet_length_max': random.randint(0, 1500),  # Packet lớn nhất backward
            'bwd_packet_length_min': random.randint(0, 100),  # Packet nhỏ nhất backward
            'bwd_packet_length_mean': random.uniform(0, 1000),  # Trung bình kích thước backward
            'bwd_packet_length_std': random.uniform(0, 500),  # Độ lệch chuẩn kích thước backward

            # Rate information (tốc độ truyền)
            'flow_bytes_s': random.uniform(0, 1000000),  # Bytes per second của flow
            'flow_packets_s': random.uniform(0, 10000),  # Packets per second của flow
            # Inter-arrival time statistics (thời gian giữa các packets)
            'flow_iat_mean': random.uniform(0, 100000),  # Trung bình IAT của flow
            'flow_iat_std': random.uniform(0, 50000),  # Độ lệch chuẩn IAT
            'flow_iat_max': random.uniform(0, 1000000),  # IAT lớn nhất
            'flow_iat_min': random.uniform(0, 10000),  # IAT nhỏ nhất
            'fwd_iat_total': random.uniform(0, 1000000),
            'fwd_iat_mean': random.uniform(0, 100000),
            'fwd_iat_std': random.uniform(0, 50000),
            'fwd_iat_max': random.uniform(0, 1000000),
            'fwd_iat_min': random.uniform(0, 10000),
            'bwd_iat_total': random.uniform(0, 1000000),
            'bwd_iat_mean': random.uniform(0, 100000),
            'bwd_iat_std': random.uniform(0, 50000),
            'bwd_iat_max': random.uniform(0, 1000000),
            'bwd_iat_min': random.uniform(0, 10000),
            # TCP flags (các cờ TCP quan trọng)
            'fwd_psh_flags': random.randint(0, 10),  # Số PSH flags forward
            'bwd_psh_flags': random.randint(0, 10),  # Số PSH flags backward
            'fwd_urg_flags': random.randint(0, 10),  # Số URG flags forward
            'bwd_urg_flags': random.randint(0, 10),  # Số URG flags backward

            # Header information
            'fwd_header_length': random.randint(20, 60),  # Tổng header length forward (bytes)
            'bwd_header_length': random.randint(20, 60),  # Tổng header length backward (bytes)
            'fwd_packets_s': random.uniform(0, 10000),
            'bwd_packets_s': random.uniform(0, 10000),
            'min_packet_length': random.randint(0, 100),
            'max_packet_length': random.randint(100, 1500),
            'packet_length_mean': random.uniform(100, 1000),
            'packet_length_std': random.uniform(0, 500),
            'packet_length_variance': random.uniform(0, 250000),
            'fin_flag_count': random.randint(0, 5),
            'syn_flag_count': random.randint(0, 5),
            'rst_flag_count': random.randint(0, 5),
            'psh_flag_count': random.randint(0, 10),
            'ack_flag_count': random.randint(0, 20),
            'urg_flag_count': random.randint(0, 5),
            'cwe_flag_count': random.randint(0, 2),
            'ece_flag_count': random.randint(0, 2),
            'down_up_ratio': random.uniform(0, 10),
            'average_packet_size': random.uniform(100, 1000),
            'avg_fwd_segment_size': random.uniform(0, 1000),
            'avg_bwd_segment_size': random.uniform(0, 1000),
            'fwd_header_length.1': random.randint(20, 60),
            'fwd_avg_bytes_bulk': random.randint(0, 1000),
            'fwd_avg_packets_bulk': random.randint(0, 100),
            'fwd_avg_bulk_rate': random.uniform(0, 1000),
            'bwd_avg_bytes_bulk': random.randint(0, 1000),
            'bwd_avg_packets_bulk': random.randint(0, 100),
            'bwd_avg_bulk_rate': random.uniform(0, 1000),
            'subflow_fwd_packets': random.randint(1, 50),
            'subflow_fwd_bytes': random.randint(0, 5000),
            'subflow_bwd_packets': random.randint(1, 50),
            'subflow_bwd_bytes': random.randint(0, 5000),
            'init_win_bytes_forward': random.randint(0, 65535),
            'init_win_bytes_backward': random.randint(0, 65535),
            'act_data_pkt_fwd': random.randint(0, 50),
            'min_seg_size_forward': random.randint(20, 60),
            'active_mean': random.uniform(0, 100000),
            'active_std': random.uniform(0, 50000),
            'active_max': random.uniform(0, 1000000),
            'active_min': random.uniform(0, 10000),
            'idle_mean': random.uniform(0, 100000),
            'idle_std': random.uniform(0, 50000),
            'idle_max': random.uniform(0, 1000000),
            'idle_min': random.uniform(0, 10000),
            'label': 'Benign'  # Trong production sẽ không có label
        }

        return network_data

    def send_network_data(self, data):
        """
        Gửi dữ liệu network đến Kafka topic đích

        Args:
            data (dict): Dictionary chứa đầy đủ thông tin network flow

        Process:
        1. Sử dụng timestamp làm partition key để đảm bảo thứ tự
        2. Gửi message đến Kafka với async call
        3. Đợi xác nhận gửi thành công (blocking call)
        4. Log thông tin metadata của message đã gửi
        """
        try:
            # Tạo partition key từ timestamp để đảm bảo messages cùng flow được
            # gửi đến cùng partition (giữ thứ tự thời gian)
            key = data.get('timestamp', str(time.time()))

            # Gửi message đến Kafka (async operation)
            # Trả về Future object để track status
            future = self.producer.send(self.topic, value=data, key=key)

            # Đợi và lấy metadata khi message được gửi thành công
            # Timeout 10 giây để tránh block forever
            record_metadata = future.get(timeout=10)

            # Log thông tin chi tiết về message đã gửi
            logger.info(f"Successfully sent network data to {record_metadata.topic} "
                       f"partition {record_metadata.partition} "
                       f"offset {record_metadata.offset}")

        except Exception as e:
            # Log lỗi và để caller xử lý (không raise exception để service tiếp tục chạy)
            logger.error(f"Failed to send network data: {e}")

    def load_historical_data(self, data_file=None):
        """
        Load dữ liệu network lịch sử từ file để testing

        Hỗ trợ format CSV chứa historical network traffic data.
        Useful cho testing với dữ liệu thực tế thay vì random data.

        Args:
            data_file (str): Đường dẫn đến file dữ liệu CSV

        Returns:
            list: List các dictionary chứa network records, empty list nếu lỗi
        """
        # Kiểm tra có file được chỉ định và là CSV
        if data_file and data_file.endswith('.csv'):
            try:
                # Đọc CSV bằng pandas (xử lý header, types tự động)
                df = pd.read_csv(data_file)

                # Log số lượng records đã load
                logger.info(f"Loaded {len(df)} historical network records from {data_file}")

                # Chuyển DataFrame thành list of dicts để dễ xử lý
                return df.to_dict('records')

            except Exception as e:
                # Log lỗi và trả về empty list
                logger.error(f"Failed to load historical data from {data_file}: {e}")
                return []

        # Không có file hoặc không phải CSV
        return []

    def start_producing(self, interval=1.0, historical_data_file=None):
        """
        Bắt đầu vòng lặp thu thập và gửi dữ liệu network liên tục

        Đây là main loop của service, chạy vô thời hạn cho đến khi bị dừng.

        Args:
            interval (float): Thời gian chờ giữa các lần gửi dữ liệu (giây)
            historical_data_file (str): File CSV chứa dữ liệu lịch sử để replay
        """
        # Đánh dấu service đang chạy
        self.is_running = True
        logger.info(f"Starting network data producer for topic: {self.topic}")

        # Load dữ liệu lịch sử nếu được chỉ định (cho testing)
        historical_data = self.load_historical_data(historical_data_file)
        data_index = 0  # Index để iterate qua historical data

        try:
            # Main production loop - chạy liên tục
            while self.is_running:
                # Chọn nguồn dữ liệu: historical hoặc synthetic
                if historical_data and data_index < len(historical_data):
                    # Sử dụng historical data
                    data = historical_data[data_index].copy()  # Copy để không modify original
                    data_index += 1

                    # Cập nhật timestamp để phản ánh thời điểm gửi hiện tại
                    data['timestamp'] = datetime.now().isoformat()
                else:
                    # Tạo dữ liệu ngẫu nhiên mới
                    data = self.generate_sample_network_data()

                # Gửi dữ liệu đến Kafka
                self.send_network_data(data)

                # Chờ interval trước khi gửi record tiếp theo
                # Điều chỉnh để control throughput
                time.sleep(interval)

        except KeyboardInterrupt:
            # Xử lý Ctrl+C graceful shutdown
            logger.info("Network producer stopped by user interrupt")
        except Exception as e:
            # Log unexpected errors nhưng không crash service
            logger.error(f"Unexpected error in producer loop: {e}")
        finally:
            # Đảm bảo cleanup khi kết thúc
            self.stop()

    def stop(self):
        """
        Dừng Network Data Producer service graceful

        Process:
        1. Đặt flag is_running = False để dừng main loop
        2. Đóng Kafka producer connection
        3. Flush remaining messages trong buffer
        4. Log trạng thái dừng
        """
        # Đặt flag để dừng main loop
        self.is_running = False

        # Cleanup Kafka producer
        if self.producer:
            # Close sẽ flush tất cả pending messages
            self.producer.close()
            logger.info("Network data producer stopped and connections closed")


def main():
    """
    Main entry point cho Network Data Producer service

    Xử lý command line arguments và khởi tạo service.
    Chạy như một standalone application.
    """
    import argparse

    # Setup command line argument parser
    parser = argparse.ArgumentParser(
        description='Safenet IDS - Network Data Producer Service',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python network_data_producer.py                                    # Run with defaults
  python network_data_producer.py --interval 0.5                   # Send every 0.5s
  python network_data_producer.py --historical-data data.csv       # Use historical data
  python network_data_producer.py --kafka-servers kafka:9092       # Custom Kafka servers
        """
    )

    # Định nghĩa arguments
    parser.add_argument('--kafka-servers', default='localhost:9092',
                       help='Kafka bootstrap servers (default: localhost:9092)')
    parser.add_argument('--topic', default='raw_network_events',
                       help='Kafka topic name (default: raw_network_events)')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Interval between sending data in seconds (default: 1.0)')
    parser.add_argument('--historical-data',
                       help='Path to CSV file with historical data for testing')

    # Parse arguments
    args = parser.parse_args()

    # Tạo thư mục logs nếu chưa tồn tại
    import os
    os.makedirs('services/logs', exist_ok=True)

    # Khởi tạo Network Data Producer với cấu hình từ command line
    producer = NetworkDataProducer(
        kafka_bootstrap_servers=args.kafka_servers,
        topic=args.topic
    )

    try:
        # Log startup
        logger.info("Starting Safenet IDS Network Data Producer...")
        logger.info(f"Configuration: servers={args.kafka_servers}, topic={args.topic}, interval={args.interval}s")

        # Bắt đầu production loop
        producer.start_producing(
            interval=args.interval,
            historical_data_file=args.historical_data
        )

    except KeyboardInterrupt:
        # Graceful shutdown khi user nhấn Ctrl+C
        logger.info("Producer stopped by user interrupt")
    except Exception as e:
        # Log và exit khi gặp lỗi không mong muốn
        logger.error(f"Producer failed with error: {e}")
        exit(1)


if __name__ == '__main__':
    main()
