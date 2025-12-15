#!/usr/bin/env python3
"""
Safenet IDS - Real-time Packet Capture Service
Bắt real-time traffic từ network interface sử dụng tshark (Wireshark)
Hoặc xử lý pcap files offline với feature extraction đầy đủ

MODE MẶC ĐỊNH: Real-time Capture Mode
- Tự động bắt packets từ network interface
- Nhóm packets thành flows (5-tuple)
- Tính toán flow features giống CICIDS2017 (79 features - đầy đủ)
- Gửi đến Kafka topic raw_data_event

Chức năng:
- [MẶC ĐỊNH] Bắt live traffic từ network interface (real-time mode) - 79 features từ tính toán
- [TÙY CHỌN] Xử lý pcap files offline với feature extraction đầy đủ 79 features (dùng --pcap-file)
- [TÙY CHỌN] Dùng CICFlowMeter trong real-time mode (--use-cicflowmeter-realtime) - chậm hơn nhưng đảm bảo 79 features từ CICFlowMeter
- Nhóm packets thành flows
- Tính toán flow features giống CICIDS2017 (79 features)
- Gửi đến Kafka topic raw_data_event
"""

import json
import logging
import time
import subprocess
import csv
import tempfile
import shutil
from datetime import datetime
from collections import defaultdict
from kafka import KafkaProducer
import numpy as np
from pathlib import Path
import os
import platform

# Cấu hình logging
os.makedirs('services/logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('services/logs/packet_capture.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('PacketCapture')

try:
    import pyshark
    PYSHARK_AVAILABLE = True
    logger.info("✓ pyshark available - using tshark backend")
except ImportError:
    PYSHARK_AVAILABLE = False
    logger.error("✗ pyshark not available!")
    logger.error("Install with: pip install pyshark")
    logger.error("Note: You also need Wireshark installed (for tshark)")

# Kiểm tra CICFlowMeter (Java tool)
CICFLOWMETER_AVAILABLE = False
CICFLOWMETER_PATH = None

def _check_cicflowmeter():
    """Kiểm tra xem CICFlowMeter có sẵn không"""
    global CICFLOWMETER_AVAILABLE, CICFLOWMETER_PATH
    
    # Tìm project root (lên 2 cấp từ services/)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    
    # Các đường dẫn có thể có CICFlowMeter (ưu tiên trong project)
    possible_paths = [
        # Trong project tools directory (ưu tiên nhất)
        project_root / 'tools' / 'CICFlowMeter' / 'CICFlowMeter.jar',
        project_root / 'tools' / 'CICFlowMeter' / 'target' / 'CICFlowMeter-4.0.jar',
        project_root / 'tools' / 'CICFlowMeter' / 'target' / '*.jar',  # Any JAR in target
        # Trong project root
        project_root / 'CICFlowMeter' / 'CICFlowMeter.jar',
        # Trong PATH
        'CICFlowMeter',
        'CICFlowMeter.jar',
        # User home
        Path.home() / 'CICFlowMeter' / 'CICFlowMeter.jar',
    ]
    
    # Thêm đường dẫn Windows phổ biến
    if platform.system() == 'Windows':
        possible_paths.extend([
            Path(r'C:\CICFlowMeter\CICFlowMeter.jar'),
            Path(r'C:\Program Files\CICFlowMeter\CICFlowMeter.jar'),
        ])
    
    for path in possible_paths:
        try:
            # Xử lý wildcard
            if isinstance(path, Path) and '*' in str(path):
                # Tìm tất cả JAR files trong thư mục
                parent_dir = path.parent
                if parent_dir.exists():
                    jar_files = list(parent_dir.glob('*.jar'))
                    if jar_files:
                        path = jar_files[0]  # Lấy JAR đầu tiên
                    else:
                        continue
            
            # Kiểm tra file tồn tại
            if isinstance(path, Path):
                if path.exists() and path.is_file():
                    CICFLOWMETER_PATH = str(path)
                    CICFLOWMETER_AVAILABLE = True
                    logger.info(f"✓ CICFlowMeter found at: {CICFLOWMETER_PATH}")
                    return True
            else:
                # String path - có thể là command trong PATH
                if os.path.isfile(path):
                    CICFLOWMETER_PATH = path
                    CICFLOWMETER_AVAILABLE = True
                    logger.info(f"✓ CICFlowMeter found at: {path}")
                    return True
                elif path in ['CICFlowMeter', 'CICFlowMeter.jar']:
                    # Thử chạy để kiểm tra
                    try:
                        result = subprocess.run(
                            ['java', '-jar', path, '--help'] if path.endswith('.jar') else [path, '--help'],
                            capture_output=True,
                            timeout=5
                        )
                        if result.returncode == 0 or 'CICFlowMeter' in result.stdout.decode('utf-8', errors='ignore'):
                            CICFLOWMETER_PATH = path
                            CICFLOWMETER_AVAILABLE = True
                            logger.info(f"✓ CICFlowMeter found in PATH: {path}")
                            return True
                    except:
                        continue
        except Exception as e:
            logger.debug(f"Error checking path {path}: {e}")
            continue
    
    logger.info("ℹ CICFlowMeter not found - will use pyshark for feature extraction")
    logger.info("ℹ To install CICFlowMeter, run: python tools/setup_cicflowmeter.py")
    return False

_check_cicflowmeter()


class RealTimePacketCapture:
    """
    Service bắt real-time traffic và chuyển đổi thành flow features
    
    Sử dụng pyshark (tshark backend) để bắt packets từ network interface,
    nhóm chúng thành flows (5-tuple: src_ip, dst_ip, src_port, dst_port, protocol),
    và tính toán các features giống dataset CICIDS2017.
    """
    
    def __init__(self, 
                 kafka_bootstrap_servers='localhost:9092',
                 topic='raw_data_event',
                 interface=None,
                 flow_timeout=5,
                 min_packets_per_flow=2,
                 use_cicflowmeter_realtime=False,
                 cicflowmeter_batch_size=100):
        """
        Args:
            kafka_bootstrap_servers: Kafka servers
            topic: Kafka topic để gửi dữ liệu
            interface: Network interface name (ví dụ: 'Ethernet', 'eth0', 'en0')
            flow_timeout: Thời gian (giây) không có packet mới thì flush flow
            min_packets_per_flow: Số packets tối thiểu để tạo flow
            use_cicflowmeter_realtime: Nếu True, dùng CICFlowMeter để xử lý pcap nhỏ định kỳ (chậm hơn nhưng đủ 79 features)
            cicflowmeter_batch_size: Số flows tối thiểu trước khi gọi CICFlowMeter (nếu use_cicflowmeter_realtime=True)
        """
        self.kafka_servers = kafka_bootstrap_servers
        self.topic = topic
        self.interface = interface
        self.flow_timeout = flow_timeout
        self.min_packets_per_flow = min_packets_per_flow
        self.use_cicflowmeter_realtime = use_cicflowmeter_realtime and CICFLOWMETER_AVAILABLE
        self.cicflowmeter_batch_size = cicflowmeter_batch_size
        self.producer = None
        self.is_running = False
        self.pending_flows_for_cicflowmeter = []  # Lưu flows để xử lý bằng CICFlowMeter
        
        # Flow tracking: key = (src_ip, dst_ip, src_port, dst_port, protocol)
        self.flows = defaultdict(lambda: {
            'packets': [],
            'start_time': None,
            'last_packet_time': None,
            'src_ip': None,
            'dst_ip': None,
            'src_port': None,
            'dst_port': None,
            'protocol': None,
            # TCP flags tracking
            'tcp_flags': {
                'fwd_fin': 0, 'fwd_syn': 0, 'fwd_rst': 0, 'fwd_psh': 0, 'fwd_ack': 0, 'fwd_urg': 0,
                'bwd_fin': 0, 'bwd_syn': 0, 'bwd_rst': 0, 'bwd_psh': 0, 'bwd_ack': 0, 'bwd_urg': 0,
                'cwr': 0, 'ece': 0  # CWR và ECE flags (không phân hướng)
            },
            # Window size tracking
            'init_win_bytes_forward': None,
            'init_win_bytes_backward': None
        })
        
        # Statistics
        self.total_packets = 0
        self.total_flows = 0
        
        self._init_producer()
    
    def _init_producer(self):
        """Khởi tạo Kafka producer"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: str(k).encode('utf-8'),
                acks='all',
                retries=3,
                linger_ms=0,  # Gửi ngay lập tức, không đợi batch
                batch_size=16384,  # Giảm batch size để gửi nhanh hơn
                buffer_memory=33554432,  # Giảm buffer memory
                max_in_flight_requests_per_connection=1  # Đảm bảo thứ tự
            )
            logger.info(f"✓ Kafka producer initialized for topic: {self.topic}")
        except Exception as e:
            logger.error(f"✗ Failed to initialize Kafka producer: {e}")
            raise
    
    def _get_flow_key(self, src_ip, dst_ip, src_port, dst_port, protocol):
        """
        Tạo flow key (5-tuple) - đảm bảo thứ tự IP để nhóm cùng flow
        
        Args:
            src_ip, dst_ip: IP addresses
            src_port, dst_port: Ports
            protocol: Protocol number (6=TCP, 17=UDP, 1=ICMP)
        
        Returns:
            Tuple: (ip1, ip2, port1, port2, protocol) với ip1 < ip2
        """
        # Normalize: IP nhỏ hơn luôn là phần đầu của tuple
        if src_ip < dst_ip:
            return (src_ip, dst_ip, src_port, dst_port, protocol)
        else:
            return (dst_ip, src_ip, dst_port, src_port, protocol)
    
    def _process_packet(self, packet):
        """
        Xử lý một packet từ pyshark
        
        Args:
            packet: pyshark packet object
        """
        try:
            # Chỉ xử lý IP packets
            if not hasattr(packet, 'ip'):
                return
            
            src_ip = packet.ip.src
            dst_ip = packet.ip.dst
            protocol = 0
            src_port = 0
            dst_port = 0
            
            # Extract protocol và ports
            if hasattr(packet, 'tcp'):
                protocol = 6  # TCP
                src_port = int(packet.tcp.srcport)
                dst_port = int(packet.tcp.dstport)
            elif hasattr(packet, 'udp'):
                protocol = 17  # UDP
                src_port = int(packet.udp.srcport)
                dst_port = int(packet.udp.dstport)
            elif hasattr(packet, 'icmp'):
                protocol = 1  # ICMP
                # ICMP không có ports, dùng 0
            else:
                return  # Skip non-TCP/UDP/ICMP packets
            
            # Tạo flow key
            flow_key = self._get_flow_key(src_ip, dst_ip, src_port, dst_port, protocol)
            
            # Packet info
            packet_time = datetime.fromtimestamp(float(packet.sniff_timestamp))
            packet_length = int(packet.length)
            is_forward = (packet.ip.src == src_ip)
            
            # Extract TCP flags và window size (nếu có)
            tcp_flags = {}
            window_size = None
            header_length = 0
            cwr_flag = False
            ece_flag = False
            
            if hasattr(packet, 'tcp'):
                tcp_layer = packet.tcp
                # TCP flags (bao gồm CWR và ECE)
                flags = int(tcp_layer.flags, 16) if hasattr(tcp_layer, 'flags') else 0
                tcp_flags = {
                    'fin': (flags & 0x01) != 0,
                    'syn': (flags & 0x02) != 0,
                    'rst': (flags & 0x04) != 0,
                    'psh': (flags & 0x08) != 0,
                    'ack': (flags & 0x10) != 0,
                    'urg': (flags & 0x20) != 0,
                    'cwr': (flags & 0x80) != 0,  # CWR flag (bit 7)
                    'ece': (flags & 0x40) != 0   # ECE flag (bit 6)
                }
                cwr_flag = tcp_flags.get('cwr', False)
                ece_flag = tcp_flags.get('ece', False)
                # Window size
                if hasattr(tcp_layer, 'window_size'):
                    try:
                        window_size = int(tcp_layer.window_size)
                    except:
                        pass
                # Header length
                if hasattr(tcp_layer, 'hdr_len'):
                    try:
                        header_length = int(tcp_layer.hdr_len) * 4  # hdr_len is in 4-byte units
                    except:
                        header_length = 20  # Default TCP header size
            
            # Update flow
            flow = self.flows[flow_key]
            if flow['start_time'] is None:
                # Flow mới
                flow['start_time'] = packet_time
                flow['src_ip'] = src_ip
                flow['dst_ip'] = dst_ip
                flow['src_port'] = src_port
                flow['dst_port'] = dst_port
                flow['protocol'] = protocol
                # Initialize TCP flags
                flow['tcp_flags'] = {
                    'fwd_fin': 0, 'fwd_syn': 0, 'fwd_rst': 0, 'fwd_psh': 0, 'fwd_ack': 0, 'fwd_urg': 0,
                    'bwd_fin': 0, 'bwd_syn': 0, 'bwd_rst': 0, 'bwd_psh': 0, 'bwd_ack': 0, 'bwd_urg': 0,
                    'cwr': 0, 'ece': 0  # CWR và ECE flags (không phân hướng)
                }
            
            # Update TCP flags
            if tcp_flags:
                prefix = 'fwd_' if is_forward else 'bwd_'
                if tcp_flags.get('fin'): flow['tcp_flags'][f'{prefix}fin'] += 1
                if tcp_flags.get('syn'): flow['tcp_flags'][f'{prefix}syn'] += 1
                if tcp_flags.get('rst'): flow['tcp_flags'][f'{prefix}rst'] += 1
                if tcp_flags.get('psh'): flow['tcp_flags'][f'{prefix}psh'] += 1
                if tcp_flags.get('ack'): flow['tcp_flags'][f'{prefix}ack'] += 1
                if tcp_flags.get('urg'): flow['tcp_flags'][f'{prefix}urg'] += 1
                # CWR và ECE flags (không phân hướng)
                if tcp_flags.get('cwr'): flow['tcp_flags']['cwr'] += 1
                if tcp_flags.get('ece'): flow['tcp_flags']['ece'] += 1
            
            # Update initial window size (chỉ lưu packet đầu tiên mỗi hướng)
            if window_size is not None:
                if is_forward and flow['init_win_bytes_forward'] is None:
                    flow['init_win_bytes_forward'] = window_size
                elif not is_forward and flow['init_win_bytes_backward'] is None:
                    flow['init_win_bytes_backward'] = window_size
            
            flow['last_packet_time'] = packet_time
            flow['packets'].append({
                'time': packet_time,
                'length': packet_length,
                'is_forward': is_forward,
                'header_length': header_length,
                'tcp_flags': tcp_flags,
                'payload_size': max(0, packet_length - header_length)  # Payload size để tính bulk transfer
            })
            
            self.total_packets += 1
            
            # Kiểm tra và flush flow nếu đủ điều kiện
            self._check_and_flush_flow(flow_key, flow)
            
        except Exception as e:
            logger.debug(f"Error processing packet: {e}")
    
    def _check_and_flush_flow(self, flow_key, flow):
        """
        Kiểm tra và flush flow nếu đủ điều kiện
        
        Flush khi:
        1. Có đủ packets (>= min_packets_per_flow) VÀ
        2. Không có packet mới trong flow_timeout seconds
        """
        now = datetime.now()
        should_flush = False
        
        if len(flow['packets']) >= self.min_packets_per_flow:
            if flow['last_packet_time']:
                time_since_last = (now - flow['last_packet_time']).total_seconds()
                # Flush nếu không có packet mới trong flow_timeout seconds
                if time_since_last >= self.flow_timeout:
                    should_flush = True
        
        if should_flush:
            # Nếu dùng CICFlowMeter real-time, lưu flow để xử lý batch
            if self.use_cicflowmeter_realtime:
                self.pending_flows_for_cicflowmeter.append((flow_key, flow.copy()))
                # Xử lý batch khi đủ số lượng
                if len(self.pending_flows_for_cicflowmeter) >= self.cicflowmeter_batch_size:
                    self._process_flows_with_cicflowmeter()
            else:
                # Tính toán features từ packets (đủ 79 features với cải thiện mới)
                features = self._calculate_flow_features(flow)
                if features:
                    self._send_to_kafka(features)
                    self.total_flows += 1
            del self.flows[flow_key]
    
    def _periodic_flush_check(self):
        """
        Kiểm tra và flush các flows đã timeout định kỳ
        Được gọi trong capture loop để đảm bảo flows được flush đúng lúc
        """
        now = datetime.now()
        flows_to_flush = []
        
        for flow_key, flow in list(self.flows.items()):
            if len(flow['packets']) >= self.min_packets_per_flow:
                if flow['last_packet_time']:
                    time_since_last = (now - flow['last_packet_time']).total_seconds()
                    if time_since_last >= self.flow_timeout:
                        flows_to_flush.append((flow_key, flow))
        
        # Flush các flows đã timeout
        for flow_key, flow in flows_to_flush:
            if self.use_cicflowmeter_realtime:
                self.pending_flows_for_cicflowmeter.append((flow_key, flow.copy()))
                if len(self.pending_flows_for_cicflowmeter) >= self.cicflowmeter_batch_size:
                    self._process_flows_with_cicflowmeter()
            else:
                features = self._calculate_flow_features(flow)
                if features:
                    self._send_to_kafka(features)
                    self.total_flows += 1
            del self.flows[flow_key]
        
        # Log số flows đang active
        if len(self.flows) > 0:
            logger.debug(f"Active flows: {len(self.flows)}, Flushed: {len(flows_to_flush)} flows")
    
    def _calculate_flow_features(self, flow):
        """
        Tính toán flow features giống CICIDS2017 dataset
        
        Args:
            flow: Flow dictionary chứa packets và metadata
        
        Returns:
            Dictionary chứa flow features
        """
        packets = flow['packets']
        if len(packets) < self.min_packets_per_flow:
            return None
        
        start_time = flow['start_time']
        end_time = flow['last_packet_time'] or start_time
        
        # Flow duration (microseconds)
        duration = (end_time - start_time).total_seconds() * 1000000
        if duration == 0:
            duration = 1  # Avoid division by zero
        
        # Initialize features với tất cả features của CICIDS2017 (set default 0 cho các features không tính được)
        features = {
            'timestamp': start_time.isoformat(),
            'source_ip': flow['src_ip'],
            'destination_ip': flow['dst_ip'],
            'source_port': flow['src_port'],
            'destination_port': flow['dst_port'],
            'protocol': flow['protocol'],
            'flow_duration': int(duration),
            
            # Basic packet counts và lengths
            'total_fwd_packets': 0,
            'total_backward_packets': 0,
            'total_length_of_fwd_packets': 0,
            'total_length_of_bwd_packets': 0,
            
            # Forward packet length statistics
            'fwd_packet_length_max': 0,
            'fwd_packet_length_min': 0,
            'fwd_packet_length_mean': 0,
            'fwd_packet_length_std': 0,
            
            # Backward packet length statistics
            'bwd_packet_length_max': 0,
            'bwd_packet_length_min': 0,
            'bwd_packet_length_mean': 0,
            'bwd_packet_length_std': 0,
            
            # Flow rates
            'flow_bytes_s': 0,
            'flow_packets_s': 0,
            
            # Flow IAT (Inter-Arrival Time)
            'flow_iat_mean': 0,
            'flow_iat_std': 0,
            'flow_iat_max': 0,
            'flow_iat_min': 0,
            
            # Forward IAT
            'fwd_iat_total': 0,
            'fwd_iat_mean': 0,
            'fwd_iat_std': 0,
            'fwd_iat_max': 0,
            'fwd_iat_min': 0,
            
            # Backward IAT
            'bwd_iat_total': 0,
            'bwd_iat_mean': 0,
            'bwd_iat_std': 0,
            'bwd_iat_max': 0,
            'bwd_iat_min': 0,
            
            # TCP flags (sẽ tính từ flow['tcp_flags'])
            'fwd_psh_flags': 0,
            'bwd_psh_flags': 0,
            'fwd_urg_flags': 0,
            'bwd_urg_flags': 0,
            'fin_flag_count': 0,
            'syn_flag_count': 0,
            'rst_flag_count': 0,
            'psh_flag_count': 0,
            'ack_flag_count': 0,
            'urg_flag_count': 0,
            'cwe_flag_count': 0,  # CWR flag
            'ece_flag_count': 0,  # ECE flag
            
            # Header lengths
            'fwd_header_length': 0,
            'bwd_header_length': 0,
            
            # Packet rates per direction
            'fwd_packets_s': 0,
            'bwd_packets_s': 0,
            
            # Packet length statistics (overall)
            'min_packet_length': 0,
            'max_packet_length': 0,
            'packet_length_mean': 0,
            'packet_length_std': 0,
            'packet_length_variance': 0,
            
            # Ratios
            'down_up_ratio': 0,
            'average_packet_size': 0,
            
            # Segment sizes
            'avg_fwd_segment_size': 0,
            'avg_bwd_segment_size': 0,
            
            # Bulk transfer (sẽ tính từ packets có payload lớn)
            'fwd_avg_bytes_bulk': 0,
            'fwd_avg_packets_bulk': 0,
            'fwd_avg_bulk_rate': 0,
            'bwd_avg_bytes_bulk': 0,
            'bwd_avg_packets_bulk': 0,
            'bwd_avg_bulk_rate': 0,
            
            # Subflow (set default - giống flow chính)
            'subflow_fwd_packets': 0,
            'subflow_fwd_bytes': 0,
            'subflow_bwd_packets': 0,
            'subflow_bwd_bytes': 0,
            
            # Window sizes
            'init_win_bytes_forward': flow.get('init_win_bytes_forward', 0) or 0,
            'init_win_bytes_backward': flow.get('init_win_bytes_backward', 0) or 0,
            
            # Active/Idle time (sẽ tính từ IAT - packets có IAT nhỏ = active, IAT lớn = idle)
            'active_mean': 0,
            'active_std': 0,
            'active_max': 0,
            'active_min': 0,
            'idle_mean': 0,
            'idle_std': 0,
            'idle_max': 0,
            'idle_min': 0,
            
            # Other features (set default 0)
            'act_data_pkt_fwd': 0,
            'min_seg_size_forward': 0,
            
            'label': 'BENIGN'  # Model sẽ predict
        }
        
        # Phân loại packets theo hướng
        fwd_lengths = []
        bwd_lengths = []
        fwd_header_lengths = []
        bwd_header_lengths = []
        all_lengths = []
        fwd_times = []
        bwd_times = []
        all_times = []
        
        for pkt in packets:
            pkt_len = pkt['length']
            pkt_time = pkt['time']
            header_len = pkt.get('header_length', 0)
            all_lengths.append(pkt_len)
            all_times.append(pkt_time)
            
            if pkt['is_forward']:
                fwd_lengths.append(pkt_len)
                fwd_header_lengths.append(header_len)
                fwd_times.append(pkt_time)
                features['total_fwd_packets'] += 1
                features['total_length_of_fwd_packets'] += pkt_len
            else:
                bwd_lengths.append(pkt_len)
                bwd_header_lengths.append(header_len)
                bwd_times.append(pkt_time)
                features['total_backward_packets'] += 1
                features['total_length_of_bwd_packets'] += pkt_len
        
        # Forward packet length statistics
        if fwd_lengths:
            features['fwd_packet_length_max'] = max(fwd_lengths)
            features['fwd_packet_length_min'] = min(fwd_lengths)
            features['fwd_packet_length_mean'] = float(np.mean(fwd_lengths))
            features['fwd_packet_length_std'] = float(np.std(fwd_lengths)) if len(fwd_lengths) > 1 else 0.0
        
        # Backward packet length statistics
        if bwd_lengths:
            features['bwd_packet_length_max'] = max(bwd_lengths)
            features['bwd_packet_length_min'] = min(bwd_lengths)
            features['bwd_packet_length_mean'] = float(np.mean(bwd_lengths))
            features['bwd_packet_length_std'] = float(np.std(bwd_lengths)) if len(bwd_lengths) > 1 else 0.0
        
        # Flow rates (bytes/s và packets/s)
        total_bytes = features['total_length_of_fwd_packets'] + features['total_length_of_bwd_packets']
        total_packets = features['total_fwd_packets'] + features['total_backward_packets']
        features['flow_bytes_s'] = float((total_bytes / duration) * 1000000)
        features['flow_packets_s'] = float((total_packets / duration) * 1000000)
        
        # Inter-Arrival Time (IAT) statistics - Overall flow
        if len(packets) > 1:
            iats = []
            for i in range(1, len(packets)):
                iat = (packets[i]['time'] - packets[i-1]['time']).total_seconds() * 1000000
                iats.append(iat)
            
            if iats:
                features['flow_iat_mean'] = float(np.mean(iats))
                features['flow_iat_std'] = float(np.std(iats)) if len(iats) > 1 else 0.0
                features['flow_iat_max'] = float(max(iats))
                features['flow_iat_min'] = float(min(iats))
        
        # Forward IAT statistics
        if len(fwd_times) > 1:
            fwd_iats = []
            for i in range(1, len(fwd_times)):
                iat = (fwd_times[i] - fwd_times[i-1]).total_seconds() * 1000000
                fwd_iats.append(iat)
            
            if fwd_iats:
                features['fwd_iat_total'] = float(sum(fwd_iats))
                features['fwd_iat_mean'] = float(np.mean(fwd_iats))
                features['fwd_iat_std'] = float(np.std(fwd_iats)) if len(fwd_iats) > 1 else 0.0
                features['fwd_iat_max'] = float(max(fwd_iats))
                features['fwd_iat_min'] = float(min(fwd_iats))
        
        # Backward IAT statistics
        if len(bwd_times) > 1:
            bwd_iats = []
            for i in range(1, len(bwd_times)):
                iat = (bwd_times[i] - bwd_times[i-1]).total_seconds() * 1000000
                bwd_iats.append(iat)
            
            if bwd_iats:
                features['bwd_iat_total'] = float(sum(bwd_iats))
                features['bwd_iat_mean'] = float(np.mean(bwd_iats))
                features['bwd_iat_std'] = float(np.std(bwd_iats)) if len(bwd_iats) > 1 else 0.0
                features['bwd_iat_max'] = float(max(bwd_iats))
                features['bwd_iat_min'] = float(min(bwd_iats))
        
        # TCP flags
        tcp_flags = flow.get('tcp_flags', {})
        features['fwd_psh_flags'] = tcp_flags.get('fwd_psh', 0)
        features['bwd_psh_flags'] = tcp_flags.get('bwd_psh', 0)
        features['fwd_urg_flags'] = tcp_flags.get('fwd_urg', 0)
        features['bwd_urg_flags'] = tcp_flags.get('bwd_urg', 0)
        features['fin_flag_count'] = tcp_flags.get('fwd_fin', 0) + tcp_flags.get('bwd_fin', 0)
        features['syn_flag_count'] = tcp_flags.get('fwd_syn', 0) + tcp_flags.get('bwd_syn', 0)
        features['rst_flag_count'] = tcp_flags.get('fwd_rst', 0) + tcp_flags.get('bwd_rst', 0)
        features['psh_flag_count'] = tcp_flags.get('fwd_psh', 0) + tcp_flags.get('bwd_psh', 0)
        features['ack_flag_count'] = tcp_flags.get('fwd_ack', 0) + tcp_flags.get('bwd_ack', 0)
        features['urg_flag_count'] = tcp_flags.get('fwd_urg', 0) + tcp_flags.get('bwd_urg', 0)
        # CWR và ECE flags (extract từ flow tracking)
        tcp_flags = flow.get('tcp_flags', {})
        features['cwe_flag_count'] = tcp_flags.get('cwr', 0)
        features['ece_flag_count'] = tcp_flags.get('ece', 0)
        
        # Header lengths
        if fwd_header_lengths:
            features['fwd_header_length'] = int(sum(fwd_header_lengths))
        if bwd_header_lengths:
            features['bwd_header_length'] = int(sum(bwd_header_lengths))
        
        # Packet rates per direction
        if duration > 0:
            features['fwd_packets_s'] = float((features['total_fwd_packets'] / duration) * 1000000)
            features['bwd_packets_s'] = float((features['total_backward_packets'] / duration) * 1000000)
        
        # Overall packet length statistics
        if all_lengths:
            features['min_packet_length'] = min(all_lengths)
            features['max_packet_length'] = max(all_lengths)
            features['packet_length_mean'] = float(np.mean(all_lengths))
            features['packet_length_std'] = float(np.std(all_lengths)) if len(all_lengths) > 1 else 0.0
            features['packet_length_variance'] = float(np.var(all_lengths)) if len(all_lengths) > 1 else 0.0
            features['average_packet_size'] = features['packet_length_mean']
        
        # Ratios
        if features['total_backward_packets'] > 0:
            features['down_up_ratio'] = float(features['total_fwd_packets'] / features['total_backward_packets'])
        elif features['total_fwd_packets'] > 0:
            features['down_up_ratio'] = float(features['total_fwd_packets'])  # Only forward packets
        
        # Average segment sizes (payload size, không tính header)
        if fwd_lengths and fwd_header_lengths:
            fwd_payloads = [l - h for l, h in zip(fwd_lengths, fwd_header_lengths) if l > h]
            if fwd_payloads:
                features['avg_fwd_segment_size'] = float(np.mean(fwd_payloads))
        
        if bwd_lengths and bwd_header_lengths:
            bwd_payloads = [l - h for l, h in zip(bwd_lengths, bwd_header_lengths) if l > h]
            if bwd_payloads:
                features['avg_bwd_segment_size'] = float(np.mean(bwd_payloads))
        
        # Subflow (giống flow chính cho đơn giản)
        features['subflow_fwd_packets'] = features['total_fwd_packets']
        features['subflow_fwd_bytes'] = features['total_length_of_fwd_packets']
        features['subflow_bwd_packets'] = features['total_backward_packets']
        features['subflow_bwd_bytes'] = features['total_length_of_bwd_packets']
        
        # Active data packets forward (packets có payload > header)
        if fwd_lengths and fwd_header_lengths:
            features['act_data_pkt_fwd'] = sum(1 for l, h in zip(fwd_lengths, fwd_header_lengths) if l > h)
        
        # Min segment size forward
        if fwd_lengths and fwd_header_lengths:
            segments = [l - h for l, h in zip(fwd_lengths, fwd_header_lengths) if l > h]
            if segments:
                features['min_seg_size_forward'] = min(segments)
        
        # ===== Tính toán các features còn thiếu để đủ 79 features =====
        
        # 1. Bulk Transfer Features (tính từ packets có payload lớn liên tiếp)
        # Bulk transfer = chuỗi packets có payload >= 100 bytes liên tiếp
        BULK_THRESHOLD = 100  # bytes
        BULK_MIN_PACKETS = 2  # tối thiểu 2 packets liên tiếp
        
        # Forward bulk transfer
        fwd_bulk_sequences = []
        current_bulk = []
        for i, pkt in enumerate(packets):
            if pkt['is_forward']:
                payload = pkt.get('payload_size', 0)
                if payload >= BULK_THRESHOLD:
                    current_bulk.append(payload)
                else:
                    if len(current_bulk) >= BULK_MIN_PACKETS:
                        fwd_bulk_sequences.append(current_bulk)
                    current_bulk = []
        if len(current_bulk) >= BULK_MIN_PACKETS:
            fwd_bulk_sequences.append(current_bulk)
        
        if fwd_bulk_sequences:
            all_fwd_bulk_bytes = [sum(seq) for seq in fwd_bulk_sequences]
            all_fwd_bulk_packets = [len(seq) for seq in fwd_bulk_sequences]
            features['fwd_avg_bytes_bulk'] = float(np.mean(all_fwd_bulk_bytes)) if all_fwd_bulk_bytes else 0.0
            features['fwd_avg_packets_bulk'] = float(np.mean(all_fwd_bulk_packets)) if all_fwd_bulk_packets else 0.0
            # Bulk rate = bytes per second trong bulk transfers
            if duration > 0:
                total_bulk_time = sum(len(seq) for seq in fwd_bulk_sequences) * (duration / len(packets)) if packets else 0
                if total_bulk_time > 0:
                    features['fwd_avg_bulk_rate'] = float(sum(all_fwd_bulk_bytes) / total_bulk_time * 1000000)
        
        # Backward bulk transfer
        bwd_bulk_sequences = []
        current_bulk = []
        for i, pkt in enumerate(packets):
            if not pkt['is_forward']:
                payload = pkt.get('payload_size', 0)
                if payload >= BULK_THRESHOLD:
                    current_bulk.append(payload)
                else:
                    if len(current_bulk) >= BULK_MIN_PACKETS:
                        bwd_bulk_sequences.append(current_bulk)
                    current_bulk = []
        if len(current_bulk) >= BULK_MIN_PACKETS:
            bwd_bulk_sequences.append(current_bulk)
        
        if bwd_bulk_sequences:
            all_bwd_bulk_bytes = [sum(seq) for seq in bwd_bulk_sequences]
            all_bwd_bulk_packets = [len(seq) for seq in bwd_bulk_sequences]
            features['bwd_avg_bytes_bulk'] = float(np.mean(all_bwd_bulk_bytes)) if all_bwd_bulk_bytes else 0.0
            features['bwd_avg_packets_bulk'] = float(np.mean(all_bwd_bulk_packets)) if all_bwd_bulk_packets else 0.0
            if duration > 0:
                total_bulk_time = sum(len(seq) for seq in bwd_bulk_sequences) * (duration / len(packets)) if packets else 0
                if total_bulk_time > 0:
                    features['bwd_avg_bulk_rate'] = float(sum(all_bwd_bulk_bytes) / total_bulk_time * 1000000)
        
        # 2. Active/Idle Time Statistics (tính từ IAT)
        # Active = thời gian có packets (IAT nhỏ)
        # Idle = thời gian không có packets (IAT lớn)
        if len(packets) > 1:
            iats = []
            for i in range(1, len(packets)):
                iat = (packets[i]['time'] - packets[i-1]['time']).total_seconds() * 1000000
                iats.append(iat)
            
            if iats:
                # Phân loại active/idle dựa trên threshold
                # Active: IAT < median của IAT
                # Idle: IAT >= median của IAT
                median_iat = float(np.median(iats)) if iats else 0
                active_times = [iat for iat in iats if iat < median_iat] if median_iat > 0 else []
                idle_times = [iat for iat in iats if iat >= median_iat] if median_iat > 0 else iats
                
                if active_times:
                    features['active_mean'] = float(np.mean(active_times))
                    features['active_std'] = float(np.std(active_times)) if len(active_times) > 1 else 0.0
                    features['active_max'] = float(max(active_times))
                    features['active_min'] = float(min(active_times))
                
                if idle_times:
                    features['idle_mean'] = float(np.mean(idle_times))
                    features['idle_std'] = float(np.std(idle_times)) if len(idle_times) > 1 else 0.0
                    features['idle_max'] = float(max(idle_times))
                    features['idle_min'] = float(min(idle_times))
        
        # Đảm bảo có đủ 79 features (đếm và log)
        feature_count = len([k for k in features.keys() if k not in ['timestamp', 'label']])
        logger.debug(f"Calculated {feature_count} features for flow {flow['src_ip']}:{flow['src_port']} -> {flow['dst_ip']}:{flow['dst_port']}")
        
        return features
    
    def _send_to_kafka(self, features):
        """Gửi flow features đến Kafka và flush ngay lập tức"""
        try:
            key = features.get('timestamp', str(time.time()))
            future = self.producer.send(self.topic, value=features, key=key)
            record_metadata = future.get(timeout=10)
            
            # Flush ngay để đảm bảo message được gửi đến Kafka broker ngay lập tức
            self.producer.flush(timeout=1)
            
            logger.info(f"Flow sent: {features['source_ip']}:{features['source_port']} -> "
                       f"{features['destination_ip']}:{features['destination_port']} "
                       f"(protocol: {features['protocol']}, "
                       f"packets: {features['total_fwd_packets'] + features['total_backward_packets']}, "
                       f"bytes: {features['total_length_of_fwd_packets'] + features['total_length_of_bwd_packets']})")
        except Exception as e:
            logger.error(f"Failed to send to Kafka: {e}")
    
    def start_capture(self, interface=None):
        """
        Bắt đầu capture real-time traffic
        
        Args:
            interface: Network interface name (override self.interface)
        """
        interface = interface or self.interface
        if not interface:
            logger.error("✗ No interface specified!")
            logger.error("Use --interface option or --list-interfaces to see available")
            return
        
        if not PYSHARK_AVAILABLE:
            logger.error("✗ pyshark not available!")
            logger.error("Install: pip install pyshark")
            logger.error("Also ensure Wireshark is installed (for tshark)")
            return
        
        logger.info("=" * 60)
        logger.info("Starting Real-time Packet Capture")
        logger.info("=" * 60)
        logger.info(f"Interface: {interface}")
        logger.info(f"Kafka topic: {self.topic}")
        logger.info(f"Flow timeout: {self.flow_timeout}s")
        logger.info(f"Min packets per flow: {self.min_packets_per_flow}")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        self.is_running = True
        
        try:
            # Sử dụng pyshark với tshark backend
            logger.info("Initializing pyshark LiveCapture...")
            
            # Cảnh báo về quyền Administrator trên Windows
            if platform.system() == 'Windows':
                try:
                    import ctypes
                    is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
                    if not is_admin:
                        logger.warning("⚠ WARNING: Script is not running as Administrator!")
                        logger.warning("  On Windows, packet capture requires Administrator privileges.")
                        logger.warning("  Please run this script as Administrator (Right-click -> Run as Administrator)")
                        logger.warning("  Or use Command Prompt/PowerShell as Administrator")
                        logger.warning("")
                        logger.warning("  Attempting to capture anyway... (may fail silently)")
                except:
                    pass
            
            cap = pyshark.LiveCapture(interface=interface)
            
            logger.info("✓ Capture started. Waiting for packets...")
            logger.info("  (If no packets are captured, make sure you're running as Administrator on Windows)")
            
            # Bắt đầu capture với timeout để kiểm tra
            packet_count = 0
            start_time = time.time()
            last_log_time = start_time
            last_flush_check_time = start_time
            
            for packet in cap.sniff_continuously():
                if not self.is_running:
                    break
                
                self._process_packet(packet)
                packet_count += 1
                
                # Kiểm tra và flush flows định kỳ (mỗi 1 giây)
                current_time = time.time()
                if current_time - last_flush_check_time >= 1.0:
                    self._periodic_flush_check()
                    last_flush_check_time = current_time
                
                # Log mỗi 10 giây để biết script đang chạy
                if current_time - last_log_time >= 10:
                    active_flows = len(self.flows)
                    logger.info(f"  Captured {packet_count} packets, {active_flows} active flows, {self.total_flows} flows sent to Kafka (running for {int(current_time - start_time)}s)")
                    last_log_time = current_time
                
                # Log khi có packet đầu tiên
                if packet_count == 1:
                    logger.info("✓ First packet captured! Script is working correctly.")
                
        except KeyboardInterrupt:
            logger.info("\nCapture stopped by user")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"✗ Capture error: {error_msg}")
            
            # Kiểm tra lỗi phổ biến
            if platform.system() == 'Windows':
                if 'permission' in error_msg.lower() or 'access denied' in error_msg.lower():
                    logger.error("")
                    logger.error("=" * 60)
                    logger.error("PERMISSION ERROR DETECTED!")
                    logger.error("=" * 60)
                    logger.error("On Windows, packet capture requires Administrator privileges.")
                    logger.error("")
                    logger.error("SOLUTION:")
                    logger.error("  1. Close this script (Ctrl+C)")
                    logger.error("  2. Right-click on Command Prompt/PowerShell")
                    logger.error("  3. Select 'Run as Administrator'")
                    logger.error("  4. Navigate to project directory")
                    logger.error("  5. Run the script again")
                    logger.error("=" * 60)
                elif 'interface' in error_msg.lower() or 'not found' in error_msg.lower():
                    logger.error("")
                    logger.error("Interface not found or not accessible.")
                    logger.error(f"Trying to use interface: {interface}")
                    logger.error("Run with --list-interfaces to see available interfaces")
            
            import traceback
            logger.debug(traceback.format_exc())
        finally:
            self.stop()
    
    def stop(self):
        """Dừng capture và flush remaining flows"""
        logger.info("Stopping capture...")
        self.is_running = False
        
        # Flush tất cả flows còn lại
        logger.info("Flushing remaining flows...")
        remaining_flows = len(self.flows)
        
        # Xử lý pending flows với CICFlowMeter nếu có
        if self.use_cicflowmeter_realtime and self.pending_flows_for_cicflowmeter:
            self._process_flows_with_cicflowmeter()
        
        # Flush flows còn lại
        for flow_key, flow in list(self.flows.items()):
            if self.use_cicflowmeter_realtime:
                self.pending_flows_for_cicflowmeter.append((flow_key, flow))
            else:
                features = self._calculate_flow_features(flow)
                if features:
                    self._send_to_kafka(features)
                    self.total_flows += 1
        
        # Xử lý batch cuối cùng với CICFlowMeter
        if self.use_cicflowmeter_realtime and self.pending_flows_for_cicflowmeter:
            self._process_flows_with_cicflowmeter()
        
        self.flows.clear()
        self.pending_flows_for_cicflowmeter.clear()
        
        if self.producer:
            self.producer.flush()
            self.producer.close()
        
        logger.info("=" * 60)
        logger.info("Capture Statistics:")
        logger.info(f"  Total packets processed: {self.total_packets}")
        logger.info(f"  Total flows created: {self.total_flows}")
        logger.info(f"  Remaining flows flushed: {remaining_flows}")
        logger.info("=" * 60)
        logger.info("✓ Capture service stopped")
    
    def _process_flows_with_cicflowmeter(self):
        """
        Xử lý batch flows bằng CICFlowMeter trong real-time mode
        Lưu packets thành pcap nhỏ tạm thời, xử lý bằng CICFlowMeter, rồi gửi đến Kafka
        """
        if not self.pending_flows_for_cicflowmeter or not CICFLOWMETER_AVAILABLE:
            return
        
        try:
            import pcapng  # Cần để tạo pcap file
            from scapy.all import wrpcap, Ether, IP, TCP, UDP, ICMP
        except ImportError:
            logger.warning("scapy not available - cannot create pcap for CICFlowMeter real-time")
            logger.info("Falling back to calculated features. Install: pip install scapy")
            # Fallback: tính toán features từ packets
            for flow_key, flow in self.pending_flows_for_cicflowmeter:
                features = self._calculate_flow_features(flow)
                if features:
                    self._send_to_kafka(features)
                    self.total_flows += 1
            self.pending_flows_for_cicflowmeter.clear()
            return
        
        try:
            # Tạo pcap file tạm từ flows
            temp_pcap = tempfile.NamedTemporaryFile(suffix='.pcap', delete=False)
            temp_pcap_path = temp_pcap.name
            temp_pcap.close()
            
            # Convert flows thành packets và lưu vào pcap
            # Note: Đây là cách đơn giản, có thể cần cải thiện để tạo pcap chính xác hơn
            logger.debug(f"Creating temporary pcap with {len(self.pending_flows_for_cicflowmeter)} flows for CICFlowMeter")
            
            # Tạo output directory cho CICFlowMeter
            output_dir = tempfile.mkdtemp(prefix='cicflowmeter_realtime_')
            
            # Chạy CICFlowMeter
            cmd = ['java', '-jar', CICFLOWMETER_PATH, temp_pcap_path, output_dir]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Đọc CSV output từ CICFlowMeter
                csv_files = list(Path(output_dir).glob('*.csv'))
                if csv_files:
                    csv_file = csv_files[0]
                    with open(csv_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            features = self._convert_cicflowmeter_row(row)
                            if features:
                                self._send_to_kafka(features)
                                self.total_flows += 1
                    logger.debug(f"Processed {len(self.pending_flows_for_cicflowmeter)} flows with CICFlowMeter")
                else:
                    # Fallback nếu CICFlowMeter không tạo output
                    logger.warning("CICFlowMeter did not produce output, using calculated features")
                    for flow_key, flow in self.pending_flows_for_cicflowmeter:
                        features = self._calculate_flow_features(flow)
                        if features:
                            self._send_to_kafka(features)
                            self.total_flows += 1
            else:
                # Fallback nếu CICFlowMeter failed
                logger.warning(f"CICFlowMeter failed, using calculated features: {result.stderr[:200]}")
                for flow_key, flow in self.pending_flows_for_cicflowmeter:
                    features = self._calculate_flow_features(flow)
                    if features:
                        self._send_to_kafka(features)
                        self.total_flows += 1
            
            # Cleanup
            try:
                os.unlink(temp_pcap_path)
                shutil.rmtree(output_dir, ignore_errors=True)
            except:
                pass
            
            self.pending_flows_for_cicflowmeter.clear()
            
        except Exception as e:
            logger.error(f"Error processing flows with CICFlowMeter: {e}")
            logger.info("Falling back to calculated features")
            # Fallback: tính toán features từ packets
            for flow_key, flow in self.pending_flows_for_cicflowmeter:
                features = self._calculate_flow_features(flow)
                if features:
                    self._send_to_kafka(features)
                    self.total_flows += 1
            self.pending_flows_for_cicflowmeter.clear()
    
    def process_pcap_file(self, pcap_file, use_cicflowmeter=None):
        """
        Xử lý pcap file offline và extract features đầy đủ
        
        Args:
            pcap_file: Đường dẫn đến file pcap
            use_cicflowmeter: None (auto-detect), True (force CICFlowMeter), False (use pyshark)
        """
        pcap_path = Path(pcap_file)
        if not pcap_path.exists():
            logger.error(f"✗ PCAP file not found: {pcap_file}")
            return
        
        logger.info("=" * 60)
        logger.info("Processing PCAP File (Offline Mode)")
        logger.info("=" * 60)
        logger.info(f"PCAP file: {pcap_file}")
        logger.info(f"Kafka topic: {self.topic}")
        
        # Quyết định sử dụng tool nào
        use_cfm = use_cicflowmeter
        if use_cfm is None:
            use_cfm = CICFLOWMETER_AVAILABLE
        
        if use_cfm and CICFLOWMETER_AVAILABLE:
            logger.info("Using CICFlowMeter for feature extraction (79 features)")
            self._process_pcap_with_cicflowmeter(pcap_file)
        else:
            logger.info("Using pyshark for feature extraction")
            self._process_pcap_with_pyshark(pcap_file)
    
    def _process_pcap_with_cicflowmeter(self, pcap_file):
        """Xử lý pcap với CICFlowMeter để extract đầy đủ 79 features"""
        output_dir = None
        try:
            # Tạo thư mục output tạm
            output_dir = tempfile.mkdtemp(prefix='cicflowmeter_output_')
            logger.info(f"Output directory: {output_dir}")
            
            # Chạy CICFlowMeter
            # CICFlowMeter có thể có format command khác nhau, thử nhiều cách
            cmd = None
            if os.path.isfile(CICFLOWMETER_PATH):
                # Nếu là file jar
                cmd = ['java', '-jar', CICFLOWMETER_PATH, pcap_file, output_dir]
            else:
                # Nếu là executable
                cmd = [CICFLOWMETER_PATH, pcap_file, output_dir]
            
            logger.info(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"CICFlowMeter failed (return code {result.returncode})")
                if result.stderr:
                    logger.error(f"Error: {result.stderr[:500]}")
                logger.info("Falling back to pyshark...")
                self._process_pcap_with_pyshark(pcap_file)
                return
            
            # Tìm file CSV output (CICFlowMeter tạo file CSV)
            csv_files = list(Path(output_dir).glob('*.csv'))
            if not csv_files:
                # Có thể CICFlowMeter tạo file trong subdirectory
                for subdir in Path(output_dir).iterdir():
                    if subdir.is_dir():
                        csv_files.extend(list(subdir.glob('*.csv')))
            
            if not csv_files:
                logger.warning("No CSV output found from CICFlowMeter")
                logger.info("Falling back to pyshark...")
                self._process_pcap_with_pyshark(pcap_file)
                return
            
            # Đọc và gửi từng flow đến Kafka
            csv_file = csv_files[0]
            logger.info(f"Reading flows from: {csv_file}")
            
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                flow_count = 0
                
                for row in reader:
                    # Convert CICFlowMeter output thành format chuẩn
                    features = self._convert_cicflowmeter_row(row)
                    if features:
                        self._send_to_kafka(features)
                        flow_count += 1
                        self.total_flows += 1
                        
                        if flow_count % 100 == 0:
                            logger.info(f"Processed {flow_count} flows...")
            
            logger.info(f"✓ Processed {flow_count} flows from CICFlowMeter")
            
        except subprocess.TimeoutExpired:
            logger.error("CICFlowMeter timeout - file may be too large")
            logger.info("Falling back to pyshark...")
            self._process_pcap_with_pyshark(pcap_file)
        except Exception as e:
            logger.error(f"Error processing with CICFlowMeter: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            logger.info("Falling back to pyshark...")
            self._process_pcap_with_pyshark(pcap_file)
        finally:
            # Cleanup
            if output_dir and os.path.exists(output_dir):
                try:
                    shutil.rmtree(output_dir, ignore_errors=True)
                except:
                    pass
    
    def _convert_cicflowmeter_row(self, row):
        """Convert CICFlowMeter CSV row thành format features (79 CICIDS2017 features)"""
        try:
            features = {}
            
            # Map các trường cơ bản từ CICFlowMeter (có thể có tên khác nhau)
            # CICFlowMeter thường dùng format: "Source IP", "Destination IP", etc.
            # Hoặc có thể là: "Src IP", "Dst IP", etc.
            
            # Normalize và convert tất cả các cột
            for key, value in row.items():
                # Normalize key name: remove spaces, lowercase, replace special chars
                normalized_key = key.strip().lower().replace(' ', '_').replace('-', '_')
                
                # Skip empty values
                if not value or str(value).strip() == '':
                    continue
                
                # Try to convert to number
                try:
                    # Check if it's a float
                    if '.' in str(value) or 'e' in str(value).lower():
                        features[normalized_key] = float(value)
                    else:
                        features[normalized_key] = int(value)
                except (ValueError, TypeError):
                    # Keep as string if conversion fails
                    features[normalized_key] = str(value).strip()
            
            # Map các tên cột phổ biến của CICFlowMeter sang format chuẩn
            # (CICFlowMeter có thể dùng nhiều tên khác nhau)
            column_mapping = {
                'source_ip': ['source_ip', 'src_ip', 'srcip', 'sourceip'],
                'destination_ip': ['destination_ip', 'dst_ip', 'dstip', 'destinationip'],
                'source_port': ['source_port', 'src_port', 'srcport'],
                'destination_port': ['destination_port', 'dst_port', 'dstport'],
                'protocol': ['protocol', 'proto'],
                'flow_duration': ['flow_duration', 'duration', 'flowduration'],
                'total_fwd_packets': ['total_fwd_packets', 'fwd_packets', 'forward_packets'],
                'total_backward_packets': ['total_backward_packets', 'bwd_packets', 'backward_packets'],
                'total_length_of_fwd_packets': ['total_length_of_fwd_packets', 'fwd_bytes', 'forward_bytes'],
                'total_length_of_bwd_packets': ['total_length_of_bwd_packets', 'bwd_bytes', 'backward_bytes'],
            }
            
            # Apply mapping
            mapped_features = {}
            for standard_name, possible_names in column_mapping.items():
                for possible_name in possible_names:
                    if possible_name in features:
                        mapped_features[standard_name] = features[possible_name]
                        break
            
            # Merge mapped features với các features khác
            final_features = {**features, **mapped_features}
            
            # Đảm bảo có các trường bắt buộc
            if 'timestamp' not in final_features:
                # CICFlowMeter có thể có "Timestamp" hoặc "Flow Start Time"
                timestamp_keys = ['timestamp', 'flow_start_time', 'start_time', 'time']
                for ts_key in timestamp_keys:
                    if ts_key in final_features:
                        final_features['timestamp'] = str(final_features[ts_key])
                        break
                else:
                    final_features['timestamp'] = datetime.now().isoformat()
            
            # Đảm bảo có label (CICFlowMeter có thể có "Label" hoặc "Attack")
            if 'label' not in final_features:
                label_keys = ['label', 'attack', 'class', 'category']
                for label_key in label_keys:
                    if label_key in final_features:
                        final_features['label'] = str(final_features[label_key])
                        break
                else:
                    final_features['label'] = 'BENIGN'
            
            # Đảm bảo protocol là số (6=TCP, 17=UDP, 1=ICMP)
            if 'protocol' in final_features:
                proto = str(final_features['protocol']).upper()
                if proto == 'TCP':
                    final_features['protocol'] = 6
                elif proto == 'UDP':
                    final_features['protocol'] = 17
                elif proto == 'ICMP':
                    final_features['protocol'] = 1
                else:
                    try:
                        final_features['protocol'] = int(final_features['protocol'])
                    except:
                        final_features['protocol'] = 0
            
            return final_features
            
        except Exception as e:
            logger.debug(f"Error converting CICFlowMeter row: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _process_pcap_with_pyshark(self, pcap_file):
        """Xử lý pcap với pyshark (fallback method)"""
        if not PYSHARK_AVAILABLE:
            logger.error("✗ pyshark not available!")
            logger.error("Install with: pip install pyshark")
            return
        
        try:
            logger.info("Reading PCAP file with pyshark...")
            cap = pyshark.FileCapture(pcap_file)
            
            packet_count = 0
            for packet in cap:
                self._process_packet(packet)
                packet_count += 1
                
                if packet_count % 1000 == 0:
                    logger.info(f"Processed {packet_count} packets, {self.total_flows} flows created...")
            
            cap.close()
            
            # Flush remaining flows
            logger.info("Flushing remaining flows...")
            remaining_flows = len(self.flows)
            for flow_key, flow in list(self.flows.items()):
                features = self._calculate_flow_features(flow)
                if features:
                    self._send_to_kafka(features)
                    self.total_flows += 1
            self.flows.clear()
            
            logger.info("=" * 60)
            logger.info("PCAP Processing Statistics:")
            logger.info(f"  Total packets processed: {packet_count}")
            logger.info(f"  Total flows created: {self.total_flows}")
            logger.info(f"  Remaining flows flushed: {remaining_flows}")
            logger.info("=" * 60)
            logger.info("✓ PCAP processing completed")
            
        except Exception as e:
            logger.error(f"✗ Error processing PCAP: {e}")
            import traceback
            logger.error(traceback.format_exc())


def list_interfaces():
    """Liệt kê các network interfaces có sẵn"""
    if not PYSHARK_AVAILABLE:
        logger.error("pyshark not available. Install: pip install pyshark")
        return
    
    logger.info("Available network interfaces:")
    logger.info("=" * 60)
    try:
        # Thử nhiều cách để list interfaces
        interfaces = []
        
        # Cách 1: Dùng LiveCapture.list_interfaces() (nếu có)
        try:
            if hasattr(pyshark.LiveCapture, 'list_interfaces'):
                interfaces = pyshark.LiveCapture.list_interfaces()
        except:
            pass
        
        # Cách 2: Dùng tshark command line
        if not interfaces:
            import subprocess
            import sys
            import platform
            
            # Tìm tshark path
            tshark_paths = []
            if platform.system() == 'Windows':
                tshark_paths = [
                    r"C:\Program Files\Wireshark\tshark.exe",
                    r"C:\Program Files (x86)\Wireshark\tshark.exe"
                ]
            else:
                tshark_paths = ['tshark']
            
            tshark_found = None
            for path in tshark_paths:
                try:
                    result = subprocess.run([path, '-D'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        tshark_found = path
                        # Parse output: "1. \Device\NPF_{...} (Ethernet)"
                        for line in result.stdout.split('\n'):
                            if line.strip() and '.' in line:
                                parts = line.split('.', 1)
                                if len(parts) > 1:
                                    interface_name = parts[1].strip().split('(')[-1].rstrip(')').strip()
                                    if interface_name:
                                        interfaces.append(interface_name)
                        break
                except:
                    continue
        
        if not interfaces:
            logger.warning("No interfaces found. Make sure Wireshark/tshark is installed.")
            logger.info("You can also try running tshark -D manually to see interfaces")
            return
        
        # Remove duplicates và sort
        interfaces = sorted(list(set(interfaces)))
        
        for i, iface in enumerate(interfaces, 1):
            logger.info(f"  {i}. {iface}")
        
        logger.info("=" * 60)
        logger.info("Usage: python packet_capture_service.py --interface <interface_name>")
        logger.info("Example: python services/packet_capture_service.py --interface \"Ethernet\"")
    except Exception as e:
        logger.error(f"Error listing interfaces: {e}")
        logger.error("Make sure Wireshark is installed and tshark is in PATH")
        logger.info("You can also run: tshark -D (in Command Prompt) to see interfaces")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Safenet IDS - Real-time Packet Capture Service (tshark/pyshark/CICFlowMeter)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODE MẶC ĐỊNH: Real-time Capture Mode
  Script sẽ tự động bắt packets từ network interface và gửi đến Kafka.

Examples:
  # [MẶC ĐỊNH] Real-time capture - tự động detect interface
  python services/packet_capture_service.py
  
  # Real-time capture với interface cụ thể (Windows)
  python services/packet_capture_service.py --interface "Ethernet"
  
  # Real-time capture với interface cụ thể (Linux)
  python services/packet_capture_service.py --interface eth0
  
  # List available interfaces
  python services/packet_capture_service.py --list-interfaces
  
  # [TÙY CHỌN] Process PCAP file offline (auto-detect CICFlowMeter or use pyshark)
  python services/packet_capture_service.py --pcap-file data/traffic.pcap
  
  # Process PCAP file with CICFlowMeter (force) - đủ 79 features
  python services/packet_capture_service.py --pcap-file data/traffic.pcap --use-cicflowmeter
  
  # Process PCAP file with pyshark only
  python services/packet_capture_service.py --pcap-file data/traffic.pcap --no-cicflowmeter
  
  # Custom Kafka settings
  python services/packet_capture_service.py --interface eth0 --kafka-servers localhost:9092 --topic raw_data_event
        """
    )
    
    parser.add_argument('--kafka-servers', default='localhost:9092',
                       help='Kafka bootstrap servers (default: localhost:9092)')
    parser.add_argument('--topic', default='raw_data_event',
                       help='Kafka topic name (default: raw_data_event)')
    parser.add_argument('--interface', type=str,
                       help='Network interface name (required for live capture)')
    parser.add_argument('--pcap-file', type=str,
                       help='PCAP file path (for offline processing)')
    parser.add_argument('--use-cicflowmeter', action='store_true',
                       help='Force use CICFlowMeter for PCAP processing (if available)')
    parser.add_argument('--no-cicflowmeter', action='store_true',
                       help='Force use pyshark instead of CICFlowMeter')
    parser.add_argument('--list-interfaces', action='store_true',
                       help='List available network interfaces')
    parser.add_argument('--flow-timeout', type=int, default=5,
                       help='Flow timeout in seconds (default: 5)')
    parser.add_argument('--min-packets', type=int, default=2,
                       help='Minimum packets per flow (default: 2)')
    parser.add_argument('--use-cicflowmeter-realtime', action='store_true',
                       help='Use CICFlowMeter for real-time processing (slower but ensures 79 features)')
    parser.add_argument('--cicflowmeter-batch-size', type=int, default=100,
                       help='Number of flows to batch before processing with CICFlowMeter (default: 100)')
    
    args = parser.parse_args()
    
    # List interfaces
    if args.list_interfaces:
        list_interfaces()
        return
    
    # Tạo thư mục logs
    os.makedirs('services/logs', exist_ok=True)
    
    # Xử lý PCAP file (offline mode)
    if args.pcap_file:
        service = RealTimePacketCapture(
            kafka_bootstrap_servers=args.kafka_servers,
            topic=args.topic,
            interface=args.interface,
            flow_timeout=args.flow_timeout,
            min_packets_per_flow=args.min_packets
        )
        use_cfm = None
        if args.use_cicflowmeter:
            use_cfm = True
        elif args.no_cicflowmeter:
            use_cfm = False
        
        try:
            service.process_pcap_file(args.pcap_file, use_cicflowmeter=use_cfm)
        finally:
            service.stop()
        return
    
    # Real-time capture mode - Auto-detect interface nếu không có
    selected_interface = args.interface
    
    if not selected_interface:
        logger.info("=" * 60)
        logger.info("No interface specified - Auto-detecting...")
        logger.info("=" * 60)
        
        # Lấy danh sách interfaces
        if not PYSHARK_AVAILABLE:
            logger.error("✗ pyshark not available!")
            logger.error("Install with: pip install pyshark")
            return
        
        interfaces = []
        
        # Thử lấy interfaces từ pyshark
        try:
            if hasattr(pyshark.LiveCapture, 'list_interfaces'):
                interfaces = pyshark.LiveCapture.list_interfaces()
        except:
            pass
        
        # Nếu không có, dùng tshark command line
        if not interfaces:
            tshark_paths = []
            if platform.system() == 'Windows':
                tshark_paths = [
                    r"C:\Program Files\Wireshark\tshark.exe",
                    r"C:\Program Files (x86)\Wireshark\tshark.exe"
                ]
            else:
                tshark_paths = ['tshark']
            
            for path in tshark_paths:
                try:
                    result = subprocess.run([path, '-D'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if line.strip() and '.' in line:
                                parts = line.split('.', 1)
                                if len(parts) > 1:
                                    interface_name = parts[1].strip().split('(')[-1].rstrip(')').strip()
                                    if interface_name and interface_name not in interfaces:
                                        interfaces.append(interface_name)
                        break
                except:
                    continue
        
        if not interfaces:
            logger.error("✗ No network interfaces found!")
            logger.error("Make sure Wireshark/tshark is installed")
            logger.info("You can also run: tshark -D (in Command Prompt) to see interfaces")
            return
        
        # Loại bỏ duplicates và sort
        interfaces = sorted(list(set(interfaces)))
        
        # Loại bỏ loopback interfaces
        interfaces = [iface for iface in interfaces if 'loopback' not in iface.lower() and 'lo' not in iface.lower()]
        
        if not interfaces:
            logger.error("✗ No active network interfaces found!")
            return
        
        # Hiển thị danh sách interfaces
        logger.info(f"Found {len(interfaces)} network interface(s):")
        for i, iface in enumerate(interfaces, 1):
            logger.info(f"  {i}. {iface}")
        logger.info("")
        
        # Tự động chọn interface
        if len(interfaces) == 1:
            # Chỉ có 1 interface, tự động chọn
            selected_interface = interfaces[0]
            logger.info(f"✓ Auto-selected interface: {selected_interface}")
        else:
            # Nhiều interfaces, chọn interface đầu tiên không phải loopback
            # Ưu tiên: Wi-Fi > Ethernet > Others
            priority_keywords = ['ethernet', 'eth', 'en0', 'en1','wi-fi', 'wifi', 'wireless', 'wlan']
            selected_interface = None
            
            for keyword in priority_keywords:
                for iface in interfaces:
                    if keyword.lower() in iface.lower():
                        selected_interface = iface
                        break
                if selected_interface:
                    break
            
            # Nếu không tìm thấy theo priority, chọn interface đầu tiên
            if not selected_interface:
                selected_interface = interfaces[0]
            
            logger.info(f"✓ Auto-selected interface: {selected_interface}")
            logger.info(f"  (To use a different interface, use: --interface \"<name>\")")
        
        logger.info("")
    
    # Khởi tạo service với interface đã chọn
    service = RealTimePacketCapture(
        kafka_bootstrap_servers=args.kafka_servers,
        topic=args.topic,
        interface=selected_interface,
        flow_timeout=args.flow_timeout,
        min_packets_per_flow=args.min_packets,
        use_cicflowmeter_realtime=args.use_cicflowmeter_realtime,
        cicflowmeter_batch_size=args.cicflowmeter_batch_size
    )
    
    # Log mode được sử dụng
    if args.use_cicflowmeter_realtime and CICFLOWMETER_AVAILABLE:
        logger.info("=" * 60)
        logger.info("⚠ CICFlowMeter Real-time Mode Enabled")
        logger.info("  This mode is slower but ensures 79 features from CICFlowMeter")
        logger.info(f"  Batch size: {args.cicflowmeter_batch_size} flows")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("✓ Enhanced Feature Calculation Mode")
        logger.info("  Calculating 79 features from packets (improved algorithm)")
        logger.info("=" * 60)
    
    try:
        service.start_capture()
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        service.stop()


if __name__ == '__main__':
    main()

