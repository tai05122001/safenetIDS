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

    def _create_base_network_data(self):
        """
        Tạo dữ liệu network flow cơ bản với tất cả features cần thiết
        
        Returns:
            dict: Dictionary chứa đầy đủ thông tin network flow với giá trị mặc định
        """
        timestamp = datetime.now().isoformat()

        def random_ip():
            return f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}"

        # Tạo dữ liệu network flow base với tất cả features của CICIDS2017
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

    def generate_benign_traffic(self):
        """
        Tạo dữ liệu network traffic bình thường (benign)
        
        Đặc trưng:
        - Số lượng packets cân bằng giữa forward và backward
        - Flow duration trung bình
        - Packet size đa dạng
        - Rate bình thường
        
        Returns:
            dict: Dictionary chứa dữ liệu benign traffic
        """
        data = self._create_base_network_data()
        
        # Đặc trưng benign: cân bằng, đa dạng
        data['total_fwd_packets'] = random.randint(10, 200)
        data['total_backward_packets'] = random.randint(10, 200)
        data['flow_duration'] = random.randint(10000, 1000000)
        data['flow_packets_s'] = random.uniform(10, 1000)
        data['flow_bytes_s'] = random.uniform(1000, 100000)
        data['fwd_packet_length_mean'] = random.uniform(200, 800)
        data['bwd_packet_length_mean'] = random.uniform(200, 800)
        data['down_up_ratio'] = random.uniform(0.5, 2.0)
        data['label'] = 'Benign'
        
        return data

    def generate_dos_attack(self, attack_subtype='hulk'):
        """
        Tạo dữ liệu giả lập DoS (Denial of Service) attack
        
        Các loại DoS:
        - hulk: HTTP Unbearable Load King - nhiều HTTP requests liên tục
        - goldeneye: Tương tự Hulk nhưng với keep-alive connections
        - slowloris: Gửi HTTP headers chậm để giữ connections mở
        - slowhttptest: Slow HTTP POST attack
        - heartbleed: Khai thác lỗ hổng Heartbleed trong OpenSSL
        
        Args:
            attack_subtype (str): Loại DoS attack cụ thể
            
        Returns:
            dict: Dictionary chứa dữ liệu DoS attack
        """
        data = self._create_base_network_data()
        
        # Đặc trưng chung của DoS: nhiều packets forward, ít backward
        data['protocol'] = 6  # TCP
        data['destination_port'] = random.choice([80, 443, 8080])  # HTTP/HTTPS ports
        
        if attack_subtype == 'hulk':
            # DoS Hulk: HTTP flood với nhiều requests
            data['total_fwd_packets'] = random.randint(5000, 20000)
            data['total_backward_packets'] = random.randint(1, 50)
            data['flow_duration'] = random.randint(1000, 50000)
            data['flow_packets_s'] = random.uniform(5000, 50000)
            data['fwd_packet_length_mean'] = random.uniform(100, 300)
            data['bwd_packet_length_mean'] = random.uniform(50, 200)
            data['syn_flag_count'] = random.randint(100, 1000)
            data['ack_flag_count'] = random.randint(50, 500)
            data['label'] = 'DoS Hulk'
            
        elif attack_subtype == 'goldeneye':
            # DoS GoldenEye: Tương tự Hulk nhưng với keep-alive
            data['total_fwd_packets'] = random.randint(3000, 15000)
            data['total_backward_packets'] = random.randint(10, 100)
            data['flow_duration'] = random.randint(5000, 100000)
            data['flow_packets_s'] = random.uniform(3000, 30000)
            data['fwd_packet_length_mean'] = random.uniform(150, 400)
            data['ack_flag_count'] = random.randint(100, 1000)  # Nhiều ACK do keep-alive
            data['label'] = 'DoS GoldenEye'
            
        elif attack_subtype == 'slowloris':
            # DoS slowloris: Gửi headers chậm
            data['total_fwd_packets'] = random.randint(100, 500)
            data['total_backward_packets'] = random.randint(1, 10)
            data['flow_duration'] = random.randint(100000, 10000000)  # Rất dài
            data['flow_packets_s'] = random.uniform(0.1, 5)  # Rất chậm
            data['fwd_iat_mean'] = random.uniform(100000, 1000000)  # IAT lớn
            data['fwd_packet_length_mean'] = random.uniform(50, 150)
            data['label'] = 'DoS slowloris'
            
        elif attack_subtype == 'slowhttptest':
            # DoS Slowhttptest: Slow HTTP POST
            data['total_fwd_packets'] = random.randint(200, 1000)
            data['total_backward_packets'] = random.randint(1, 20)
            data['flow_duration'] = random.randint(50000, 5000000)
            data['flow_packets_s'] = random.uniform(0.5, 10)
            data['fwd_packet_length_mean'] = random.uniform(100, 200)
            data['psh_flag_count'] = random.randint(10, 100)
            data['label'] = 'DoS Slowhttptest'
            
        elif attack_subtype == 'heartbleed':
            # Heartbleed: Khai thác lỗ hổng OpenSSL
            data['total_fwd_packets'] = random.randint(50, 500)
            data['total_backward_packets'] = random.randint(10, 100)
            data['flow_duration'] = random.randint(10000, 100000)
            data['destination_port'] = 443  # HTTPS
            data['fwd_packet_length_mean'] = random.uniform(500, 2000)  # Payload lớn
            data['bwd_packet_length_mean'] = random.uniform(1000, 10000)  # Response lớn
            data['urg_flag_count'] = random.randint(5, 50)
            data['label'] = 'Heartbleed'
        else:
            # DoS generic
            data['total_fwd_packets'] = random.randint(1000, 10000)
            data['total_backward_packets'] = random.randint(1, 100)
            data['flow_duration'] = random.randint(1000, 100000)
            data['flow_packets_s'] = random.uniform(1000, 20000)
            data['label'] = f'DoS {attack_subtype}'
        
        # Đặc trưng chung DoS
        data['down_up_ratio'] = random.uniform(50, 500)  # Rất cao (nhiều forward)
        data['flow_bytes_s'] = random.uniform(100000, 10000000)
        
        return data

    def generate_ddos_attack(self):
        """
        Tạo dữ liệu giả lập DDoS (Distributed Denial of Service) attack
        
        Đặc trưng:
        - Cực nhiều packets từ nhiều sources
        - Flow duration rất ngắn
        - Packet rate cực cao
        - Nhiều SYN flags
        
        Returns:
            dict: Dictionary chứa dữ liệu DDoS attack
        """
        data = self._create_base_network_data()
        
        # Đặc trưng DDoS: cực nhiều packets, rate cao
        data['total_fwd_packets'] = random.randint(50000, 500000)
        data['total_backward_packets'] = random.randint(1, 100)
        data['flow_duration'] = random.randint(100, 10000)  # Rất ngắn
        data['flow_packets_s'] = random.uniform(10000, 100000)  # Cực cao
        data['flow_bytes_s'] = random.uniform(10000000, 100000000)
        data['fwd_packet_length_mean'] = random.uniform(50, 200)
        data['fwd_packet_length_std'] = random.uniform(10, 50)  # Đồng nhất
        data['syn_flag_count'] = random.randint(1000, 10000)
        data['ack_flag_count'] = random.randint(100, 1000)
        data['down_up_ratio'] = random.uniform(500, 5000)
        data['protocol'] = 6  # TCP
        data['destination_port'] = random.choice([80, 443, 53])  # HTTP/HTTPS/DNS
        data['label'] = 'DDoS'
        
        return data

    def generate_bot_attack(self):
        """
        Tạo dữ liệu giả lập Bot attack
        
        Đặc trưng:
        - Nhiều connections từ cùng source
        - Pattern lặp lại
        - Medium packet rate
        - Đa dạng ports
        
        Returns:
            dict: Dictionary chứa dữ liệu Bot attack
        """
        data = self._create_base_network_data()
        
        # Đặc trưng Bot: pattern lặp lại, nhiều connections
        data['total_fwd_packets'] = random.randint(100, 1000)
        data['total_backward_packets'] = random.randint(50, 500)
        data['flow_duration'] = random.randint(10000, 1000000)
        data['flow_packets_s'] = random.uniform(10, 500)
        data['flow_bytes_s'] = random.uniform(10000, 1000000)
        data['fwd_packet_length_mean'] = random.uniform(200, 600)
        data['bwd_packet_length_mean'] = random.uniform(200, 600)
        data['down_up_ratio'] = random.uniform(0.5, 3.0)
        data['protocol'] = 6  # TCP
        data['destination_port'] = random.choice([80, 443, 21, 22, 25, 53])
        data['syn_flag_count'] = random.randint(5, 50)
        data['ack_flag_count'] = random.randint(10, 100)
        data['label'] = 'Bot'
        
        return data

    def generate_rare_attack(self, attack_subtype='sql_injection'):
        """
        Tạo dữ liệu giả lập Rare Attack (các tấn công hiếm, phức tạp)
        
        Các loại rare attack:
        - sql_injection: SQL Injection attack
        - xss: Cross-Site Scripting
        - brute_force: Brute Force attack
        - ftp_patator: FTP brute force
        - ssh_patator: SSH brute force
        - web_attack: Web application attack
        - infiltration: Infiltration attack
        
        Args:
            attack_subtype (str): Loại rare attack cụ thể
            
        Returns:
            dict: Dictionary chứa dữ liệu rare attack
        """
        data = self._create_base_network_data()
        
        if attack_subtype == 'sql_injection':
            # SQL Injection: Payload lớn, header dài
            data['total_fwd_packets'] = random.randint(10, 100)
            data['total_backward_packets'] = random.randint(5, 50)
            data['flow_duration'] = random.randint(10000, 100000)
            data['destination_port'] = random.choice([80, 443, 3306, 5432])  # HTTP/HTTPS/MySQL/PostgreSQL
            data['fwd_header_length'] = random.randint(100, 500)  # Header lớn
            data['total_length_of_fwd_packets'] = random.randint(5000, 50000)  # Payload lớn
            data['fwd_packet_length_mean'] = random.uniform(500, 2000)
            data['psh_flag_count'] = random.randint(5, 20)
            data['label'] = 'SQL Injection'
            
        elif attack_subtype == 'xss':
            # XSS: Tương tự SQL Injection nhưng nhỏ hơn
            data['total_fwd_packets'] = random.randint(5, 50)
            data['total_backward_packets'] = random.randint(5, 30)
            data['flow_duration'] = random.randint(5000, 50000)
            data['destination_port'] = random.choice([80, 443, 8080])
            data['fwd_header_length'] = random.randint(80, 300)
            data['total_length_of_fwd_packets'] = random.randint(2000, 20000)
            data['label'] = 'XSS'
            
        elif attack_subtype == 'brute_force':
            # Brute Force: Nhiều connections ngắn
            data['total_fwd_packets'] = random.randint(50, 500)
            data['total_backward_packets'] = random.randint(20, 200)
            data['flow_duration'] = random.randint(1000, 10000)  # Ngắn
            data['flow_packets_s'] = random.uniform(50, 500)
            data['syn_flag_count'] = random.randint(10, 100)
            data['rst_flag_count'] = random.randint(5, 50)  # Nhiều failed connections
            data['destination_port'] = random.choice([21, 22, 23, 80, 443])
            data['label'] = 'Brute Force'
            
        elif attack_subtype == 'ftp_patator':
            # FTP Patator: FTP brute force
            data['total_fwd_packets'] = random.randint(100, 1000)
            data['total_backward_packets'] = random.randint(50, 500)
            data['flow_duration'] = random.randint(5000, 50000)
            data['destination_port'] = 21  # FTP
            data['protocol'] = 6  # TCP
            data['syn_flag_count'] = random.randint(20, 200)
            data['ack_flag_count'] = random.randint(10, 100)
            data['label'] = 'FTP-Patator'
            
        elif attack_subtype == 'ssh_patator':
            # SSH Patator: SSH brute force
            data['total_fwd_packets'] = random.randint(50, 500)
            data['total_backward_packets'] = random.randint(30, 300)
            data['flow_duration'] = random.randint(5000, 50000)
            data['destination_port'] = 22  # SSH
            data['protocol'] = 6  # TCP
            data['syn_flag_count'] = random.randint(10, 100)
            data['rst_flag_count'] = random.randint(5, 50)
            data['label'] = 'SSH-Patator'
            
        elif attack_subtype == 'web_attack':
            # Web Attack: Tấn công web application
            data['total_fwd_packets'] = random.randint(20, 200)
            data['total_backward_packets'] = random.randint(10, 100)
            data['flow_duration'] = random.randint(10000, 100000)
            data['destination_port'] = random.choice([80, 443, 8080, 8443])
            data['fwd_header_length'] = random.randint(100, 400)
            data['total_length_of_fwd_packets'] = random.randint(3000, 30000)
            data['psh_flag_count'] = random.randint(5, 30)
            data['label'] = 'Web Attack'
            
        elif attack_subtype == 'infiltration':
            # Infiltration: Tấn công xâm nhập
            data['total_fwd_packets'] = random.randint(100, 1000)
            data['total_backward_packets'] = random.randint(50, 500)
            data['flow_duration'] = random.randint(20000, 200000)
            data['flow_packets_s'] = random.uniform(10, 200)
            data['destination_port'] = random.choice([22, 23, 80, 443, 3389])  # SSH/Telnet/HTTP/HTTPS/RDP
            data['fwd_packet_length_mean'] = random.uniform(300, 800)
            data['bwd_packet_length_mean'] = random.uniform(300, 800)
            data['label'] = 'Infiltration'
        else:
            # Generic rare attack
            data['total_fwd_packets'] = random.randint(20, 200)
            data['total_backward_packets'] = random.randint(10, 100)
            data['flow_duration'] = random.randint(10000, 100000)
            data['fwd_header_length'] = random.randint(100, 300)
            data['label'] = f'Rare Attack ({attack_subtype})'
        
        # Đặc trưng chung rare attacks
        data['protocol'] = 6  # TCP
        data['flow_bytes_s'] = random.uniform(5000, 500000)
        data['down_up_ratio'] = random.uniform(0.5, 5.0)
        
        return data

    def generate_attack_data(self, attack_type='benign', attack_subtype=None):
        """
        Tạo dữ liệu network traffic theo loại tấn công được chỉ định
        
        Args:
            attack_type (str): Loại tấn công chính:
                - 'benign': Traffic bình thường
                - 'dos': DoS attack
                - 'ddos': DDoS attack
                - 'bot': Bot attack
                - 'rare_attack': Rare attack
            attack_subtype (str): Loại tấn công cụ thể (tùy chọn):
                - DoS: 'hulk', 'goldeneye', 'slowloris', 'slowhttptest', 'heartbleed'
                - Rare: 'sql_injection', 'xss', 'brute_force', 'ftp_patator', 
                        'ssh_patator', 'web_attack', 'infiltration'
        
        Returns:
            dict: Dictionary chứa dữ liệu network traffic
        """
        if attack_type == 'benign':
            return self.generate_benign_traffic()
        elif attack_type == 'dos':
            subtype = attack_subtype or random.choice(['hulk', 'goldeneye', 'slowloris', 'slowhttptest', 'heartbleed'])
            return self.generate_dos_attack(subtype)
        elif attack_type == 'ddos':
            return self.generate_ddos_attack()
        elif attack_type == 'bot':
            return self.generate_bot_attack()
        elif attack_type == 'rare_attack':
            subtype = attack_subtype or random.choice([
                'sql_injection', 'xss', 'brute_force', 'ftp_patator', 
                'ssh_patator', 'web_attack', 'infiltration'
            ])
            return self.generate_rare_attack(subtype)
        else:
            # Mặc định trả về benign nếu không nhận dạng được
            logger.warning(f"Unknown attack type: {attack_type}, using benign")
            return self.generate_benign_traffic()

    def generate_sample_network_data(self):
        """
        Tạo mẫu dữ liệu network traffic để testing/demo (backward compatibility)
        
        Mặc định tạo benign traffic. Để tạo tấn công, sử dụng generate_attack_data()
        
        Returns:
            dict: Dictionary chứa dữ liệu network flow
        """
        return self.generate_benign_traffic()

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

    def start_producing(self, interval=1.0, historical_data_file=None, attack_type='benign', attack_subtype=None, attack_mix=None):
        """
        Bắt đầu vòng lặp thu thập và gửi dữ liệu network liên tục

        Đây là main loop của service, chạy vô thời hạn cho đến khi bị dừng.

        Args:
            interval (float): Thời gian chờ giữa các lần gửi dữ liệu (giây)
            historical_data_file (str): File CSV chứa dữ liệu lịch sử để replay
            attack_type (str): Loại tấn công để tạo ('benign', 'dos', 'ddos', 'bot', 'rare_attack')
            attack_subtype (str): Loại tấn công cụ thể (tùy chọn)
            attack_mix (dict): Dictionary chứa tỷ lệ các loại tấn công để tạo mix
                             Ví dụ: {'benign': 0.5, 'dos': 0.3, 'ddos': 0.2}
        """
        # Đánh dấu service đang chạy
        self.is_running = True
        logger.info(f"Starting network data producer for topic: {self.topic}")
        
        if attack_mix:
            logger.info(f"Using attack mix: {attack_mix}")
        else:
            logger.info(f"Generating {attack_type} attacks" + (f" ({attack_subtype})" if attack_subtype else ""))

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
                    # Tạo dữ liệu synthetic
                    if attack_mix:
                        # Chọn loại tấn công dựa trên tỷ lệ trong attack_mix
                        rand = random.random()
                        cumulative = 0.0
                        selected_type = 'benign'  # Default
                        selected_subtype = None
                        
                        for atk_type, prob in attack_mix.items():
                            cumulative += prob
                            if rand <= cumulative:
                                selected_type = atk_type
                                # Xử lý subtype nếu có (ví dụ: 'dos:hulk')
                                if ':' in atk_type:
                                    parts = atk_type.split(':')
                                    selected_type = parts[0]
                                    selected_subtype = parts[1]
                                break
                        
                        data = self.generate_attack_data(selected_type, selected_subtype)
                    else:
                        # Tạo dữ liệu theo attack_type được chỉ định
                        data = self.generate_attack_data(attack_type, attack_subtype)

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

    def start_test_mode(self, interval=1.0):
        """
        Chế độ test: Gửi 2 mẫu benign + 3 mẫu tấn công từ 3 loại khác nhau rồi dừng
        
        Thứ tự gửi:
        1. Benign traffic (2 mẫu)
        2. DoS attack (1 mẫu - hulk)
        3. DDoS attack (1 mẫu)
        4. Rare attack (1 mẫu - sql_injection)
        
        Args:
            interval (float): Thời gian chờ giữa các lần gửi dữ liệu (giây)
        """
        self.is_running = True
        logger.info("=" * 70)
        logger.info("Starting TEST MODE: 2 benign + 3 attacks from different types")
        logger.info("=" * 70)
        
        test_sequence = [
            ('benign', None, 'Benign Traffic #1'),
            ('benign', None, 'Benign Traffic #2'),
            ('dos', 'hulk', 'DoS Hulk Attack'),
            ('ddos', None, 'DDoS Attack'),
            ('rare_attack', 'sql_injection', 'SQL Injection Attack'),
        ]
        
        try:
            for i, (attack_type, attack_subtype, description) in enumerate(test_sequence, 1):
                if not self.is_running:
                    break
                    
                logger.info(f"[{i}/5] Generating: {description}")
                
                # Tạo dữ liệu theo loại tấn công
                data = self.generate_attack_data(attack_type, attack_subtype)
                
                # Gửi dữ liệu đến Kafka
                self.send_network_data(data)
                
                logger.info(f"[{i}/5] ✓ Sent: {description} (label: {data.get('label', 'N/A')})")
                
                # Chờ interval trước khi gửi mẫu tiếp theo (trừ mẫu cuối)
                if i < len(test_sequence):
                    time.sleep(interval)
            
            logger.info("=" * 70)
            logger.info("TEST MODE completed successfully!")
            logger.info(f"Total samples sent: {len(test_sequence)}")
            logger.info("  - 2 Benign traffic samples")
            logger.info("  - 1 DoS attack (Hulk)")
            logger.info("  - 1 DDoS attack")
            logger.info("  - 1 Rare attack (SQL Injection)")
            logger.info("=" * 70)
            
        except KeyboardInterrupt:
            logger.info("Test mode stopped by user interrupt")
        except Exception as e:
            logger.error(f"Error in test mode: {e}")
        finally:
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
  python network_data_producer.py                                    # Run with defaults (benign)
  python network_data_producer.py --test-mode                       # Test: 2 benign + 3 attacks
  python network_data_producer.py --test-mode --interval 0.5        # Test mode with 0.5s interval
  python network_data_producer.py --interval 0.5                   # Send every 0.5s
  python network_data_producer.py --historical-data data.csv       # Use historical data
  python network_data_producer.py --attack-type dos                # Generate DoS attacks
  python network_data_producer.py --attack-type dos --attack-subtype hulk  # DoS Hulk
  python network_data_producer.py --attack-type rare_attack --attack-subtype sql_injection  # SQL Injection
  python network_data_producer.py --attack-mix '{"benign":0.7,"dos":0.2,"ddos":0.1}'  # Mixed attacks
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
    parser.add_argument('--attack-type', 
                       choices=['benign', 'dos', 'ddos', 'bot', 'rare_attack'],
                       default='benign',
                       help='Type of attack to generate (default: benign)')
    parser.add_argument('--attack-subtype',
                       help='Specific attack subtype (e.g., hulk, sql_injection, ftp_patator)')
    parser.add_argument('--attack-mix',
                       help='Attack mix as JSON string (e.g., \'{"benign":0.5,"dos":0.3,"ddos":0.2}\')')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode: Send 2 benign + 3 attacks from different types then stop')

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
        # Parse attack mix nếu có
        attack_mix = None
        if args.attack_mix:
            try:
                attack_mix = json.loads(args.attack_mix)
                # Validate tổng tỷ lệ = 1.0
                total = sum(attack_mix.values())
                if abs(total - 1.0) > 0.01:
                    logger.warning(f"Attack mix probabilities sum to {total}, normalizing to 1.0")
                    attack_mix = {k: v/total for k, v in attack_mix.items()}
            except json.JSONDecodeError as e:
                logger.error(f"Invalid attack mix JSON: {e}")
                exit(1)

        # Log startup
        logger.info("Starting Safenet IDS Network Data Producer...")
        logger.info(f"Configuration: servers={args.kafka_servers}, topic={args.topic}, interval={args.interval}s")
        
        # Kiểm tra chế độ test
        if args.test_mode:
            # Chế độ test: gửi 2 benign + 3 attacks rồi dừng
            producer.start_test_mode(interval=args.interval)
        else:
            # Chế độ production: chạy liên tục
            if attack_mix:
                logger.info(f"Attack mix: {attack_mix}")
            else:
                logger.info(f"Attack type: {args.attack_type}" + (f" ({args.attack_subtype})" if args.attack_subtype else ""))

            # Bắt đầu production loop
            producer.start_producing(
                interval=args.interval,
                historical_data_file=args.historical_data,
                attack_type=args.attack_type,
                attack_subtype=args.attack_subtype,
                attack_mix=attack_mix
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
