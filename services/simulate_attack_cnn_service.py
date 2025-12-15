import pandas as pd
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from network_data_producer import NetworkDataProducer
import time
import random
from datetime import datetime
import logging
import os
import sys
from kafka import KafkaConsumer, KafkaAdminClient
from kafka.admin import NewPartitions
from kafka.structs import TopicPartition

# Tạo thư mục logs nếu chưa tồn tại (phải làm trước khi khởi tạo FileHandler)
os.makedirs('services/logs', exist_ok=True)

# Cố gắng set UTF-8 encoding cho console trên Windows
if sys.platform == 'win32':
    try:
        # Thử set console code page sang UTF-8
        import subprocess
        subprocess.run(['chcp', '65001'], shell=True, capture_output=True, check=False)
        # Set stdout/stderr encoding nếu có thể
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass  # Nếu không được thì bỏ qua

# Custom StreamHandler với error handling cho Windows console
class SafeStreamHandler(logging.StreamHandler):
    """StreamHandler với error handling cho encoding issues trên Windows"""
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            
            # Thử ghi trực tiếp
            try:
                stream.write(msg + self.terminator)
                self.flush()
            except (UnicodeEncodeError, UnicodeDecodeError):
                # Nếu lỗi encoding, chuyển đổi message sang ASCII-safe
                # bằng cách thay thế các ký tự không encode được bằng '?'
                safe_msg = msg.encode('ascii', errors='replace').decode('ascii', errors='replace')
                try:
                    stream.write(safe_msg + self.terminator)
                    self.flush()
                except Exception:
                    # Nếu vẫn lỗi, bỏ qua
                    self.handleError(record)
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('services/logs/simulate_attack_cnn.log', encoding='utf-8'),
        SafeStreamHandler()
    ]
)
logger = logging.getLogger('SimulateAttackCNN')

def reset_consumer_group_offset(kafka_bootstrap_servers='localhost:9092', 
                                topic='raw_data_event_cnn',
                                group_id='safenet-cnn-preprocessing-group'):
    """
    Reset offset của consumer group về cuối topic để đảm bảo chỉ đọc dữ liệu mới
    
    Args:
        kafka_bootstrap_servers: Kafka bootstrap servers
        topic: Topic name
        group_id: Consumer group ID cần reset
    """
    try:
        logger.info(f"Resetting offset for consumer group '{group_id}' on topic '{topic}'...")
        
        # Tạo consumer tạm với group_id khác để lấy partition info
        # Điều này tránh conflict với consumer đang chạy
        temp_consumer = KafkaConsumer(
            topic,
            bootstrap_servers=kafka_bootstrap_servers,
            group_id=f'{group_id}-temp-reset-{int(time.time())}',  # Group ID tạm thời unique
            enable_auto_commit=False,
            auto_offset_reset='latest',
            consumer_timeout_ms=5000  # Timeout sau 5 giây
        )
        
        # Poll để trigger partition assignment
        temp_consumer.poll(timeout_ms=2000)
        partitions = temp_consumer.assignment()
        
        if not partitions:
            logger.warning(f"Could not get partitions for topic '{topic}'")
            temp_consumer.close()
            return
        
        # Lấy end offsets cho tất cả partitions
        end_offsets = temp_consumer.end_offsets(partitions)
        logger.info(f"Found {len(partitions)} partitions, end offsets: {end_offsets}")
        
        temp_consumer.close()
        
        # Bây giờ tạo consumer với group_id thực để reset offset
        # Đợi một chút để consumer cũ có thể rejoin nếu cần
        time.sleep(1)
        
        reset_consumer = KafkaConsumer(
            topic,
            bootstrap_servers=kafka_bootstrap_servers,
            group_id=group_id,
            enable_auto_commit=False,
            auto_offset_reset='latest',
            consumer_timeout_ms=10000
        )
        
        # Poll và đợi partition assignment (có thể mất thời gian nếu consumer cũ đang chạy)
        logger.info("Waiting for partition assignment (consumer may need to rejoin group)...")
        assigned = None
        for attempt in range(15):  # Thử tối đa 15 lần (15 giây)
            reset_consumer.poll(timeout_ms=1000)
            assigned = reset_consumer.assignment()
            if assigned:
                logger.info(f"Partitions assigned: {assigned}")
                break
            if attempt < 14:  # Không sleep ở lần cuối
                time.sleep(0.5)
        
        if not assigned:
            logger.warning("Could not get partition assignment after waiting, cannot reset offset")
            reset_consumer.close()
            return
        
        # Đảm bảo partition đã được assign trước khi seek
        # Chỉ seek partition có trong assignment và có trong end_offsets
        seeked_count = 0
        for partition in assigned:
            # Kiểm tra partition có trong assignment và có trong end_offsets
            if partition in reset_consumer.assignment() and partition in end_offsets:
                try:
                    reset_consumer.seek(partition, end_offsets[partition])
                    logger.info(f"Seeked partition {partition.partition} to offset {end_offsets[partition]}")
                    seeked_count += 1
                except AssertionError as e:
                    logger.warning(f"Partition {partition.partition} not assigned, skipping: {e}")
                except Exception as e:
                    logger.warning(f"Failed to seek partition {partition.partition}: {e}")
            else:
                logger.warning(f"Partition {partition.partition} not in assignment or end_offsets, skipping")
        
        if seeked_count > 0:
            # Commit offset
            try:
                reset_consumer.commit()
                logger.info(f"Successfully reset and committed offset for consumer group '{group_id}' ({seeked_count} partitions)")
            except Exception as e:
                logger.warning(f"Failed to commit offset: {e}")
        else:
            logger.warning("No partitions were seeked, cannot reset offset")
        
        reset_consumer.close()
        logger.info("Offset reset completed")
        
    except Exception as e:
        logger.error(f"Failed to reset consumer group offset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.warning("Continuing without offset reset - consumer may read old messages")

def simulate_attack_from_pkl(num_samples=5, pkl_file='dataset.pkl', kafka_bootstrap_servers='localhost:9092', topic='raw_data_event_cnn', force_reset=True):
    """
    Tải dữ liệu từ dataset.pkl, chọn các mẫu đại diện cho tất cả loại tấn công
    để đảm bảo cả Level 1 và Level 2 model đều có thể hoạt động.

    **MẶC ĐỊNH KHI START (5 samples mỗi loại, oversample nếu cần):**
    - BENIGN: 5 samples (không tấn công)
    - DoS attacks: 20 samples (5 samples mỗi loại DoS: hulk, goldeneye, slowloris, slowhttptest)
    - DDoS: 5 samples
    - PortScan: 5 samples

    Tổng cộng: 5 + 20 + 5 + 5 = 35 samples

    Nếu dataset không có đủ mẫu cho một loại nào đó, sẽ oversample (lấy mẫu có lặp lại)
    để đảm bảo có đúng số lượng mẫu mong muốn.
    """
    try:
        df = pd.read_pickle(pkl_file)
        logger.info(f"Loaded {len(df)} records from {pkl_file}")

        # Tìm cột label (có thể có nhiều tên khác nhau)
        # Thử các tên phổ biến: Label, label, label_encoded, label_group, etc.
        label_col = None
        possible_label_cols = ['Label', 'label', 'label_encoded', 'label_group', 
                              'Label_encoded', 'Label_group', 'target', 'Target']
        
        for col in possible_label_cols:
            if col in df.columns:
                label_col = col
                logger.info(f"Found label column: '{col}'")
                break
        
        # Nếu không tìm thấy, log tất cả các cột để debug
        if label_col is None:
            logger.warning("Không tìm thấy cột label trong dataset!")
            logger.info(f"Dataset có {len(df.columns)} cột:")
            logger.info(f"Các cột: {', '.join(df.columns.tolist()[:20])}")  # Hiển thị 20 cột đầu
            if len(df.columns) > 20:
                logger.info(f"... và {len(df.columns) - 20} cột khác")
            
            # Thử tìm cột có chứa 'label' trong tên (case-insensitive)
            label_like_cols = [col for col in df.columns if 'label' in str(col).lower()]
            if label_like_cols:
                logger.info(f"Tìm thấy các cột có chứa 'label': {', '.join(label_like_cols)}")
                # Sử dụng cột đầu tiên tìm thấy
                label_col = label_like_cols[0]
                logger.info(f"Sử dụng cột '{label_col}' làm label column")
        
        # Định nghĩa các loại tấn công cần thiết (dùng cho cả khi có và không có label_col)
        # DoS attacks (cho Level 2 dos model) - chia đều các loại
        # Đã bỏ Heartbleed khỏi dataset
        dos_subtypes = {
            'hulk': ['DoS Hulk', 'dos hulk', 'DoS attack'],
            'goldeneye': ['DoS GoldenEye', 'dos goldeneye'],
            'slowloris': ['DoS slowloris', 'dos slowloris'],
            'slowhttptest': ['DoS Slowhttptest', 'dos slowhttptest']
        }
        # Tất cả DoS labels (để tìm trong dataset)
        dos_labels = []
        for subtype_labels in dos_subtypes.values():
            dos_labels.extend(subtype_labels)
        
        # Benign
        benign_labels = ['BENIGN', 'Benign', 'benign']
        
        # DDoS
        ddos_labels = ['DDoS', 'DDOS attack-HOIC', 'DDOS attack-LOIC-UDP', 
                      'ddos', 'ddos attack-hoic', 'ddos attack-loic-udp']
        
        # PortScan
        portscan_labels = ['PortScan', 'portscan', 'Port Scan', 'port scan']
        
        if label_col is None:
            logger.warning("Không tìm thấy cột label, chọn ngẫu nhiên")
            if len(df) < num_samples:
                samples_df = df
            else:
                random_indices = random.sample(range(len(df)), num_samples)
                samples_df = df.iloc[random_indices]
        else:
            selected_samples = []
            samples_per_type = 5  # Mỗi loại lấy đúng 5 samples (oversample nếu cần)
            
            # 1. Chọn BENIGN (5 samples) - Oversample nếu cần
            benign_df = df[df[label_col].isin(benign_labels)]
            if len(benign_df) > 0:
                # Oversample nếu cần để có đúng 5 mẫu
                benign_samples = benign_df.sample(n=samples_per_type, replace=True, random_state=42)
                selected_samples.append(benign_samples)
                logger.info(f"Selected {len(benign_samples)} BENIGN samples (oversampled if needed)")
            else:
                logger.warning("Không tìm thấy BENIGN samples trong dataset!")

            # 2. Chọn DoS attacks - 5 samples MỖI loại DoS (oversample nếu cần)
            for subtype_name, subtype_labels in dos_subtypes.items():
                subtype_df = df[df[label_col].isin(subtype_labels)]
                if len(subtype_df) > 0:
                    # Oversample nếu cần để có đúng 5 mẫu cho mỗi subtype
                    subtype_samples = subtype_df.sample(n=samples_per_type, replace=True, random_state=42)
                    selected_samples.append(subtype_samples)
                    logger.info(f"Selected {len(subtype_samples)} DoS {subtype_name} samples (oversampled if needed)")
                else:
                    logger.warning(f"Không tìm thấy DoS {subtype_name} samples trong dataset!")

            # 3. Chọn DDoS (5 samples) - Oversample nếu cần
            ddos_df = df[df[label_col].isin(ddos_labels)]
            if len(ddos_df) > 0:
                # Oversample nếu cần để có đúng 5 mẫu
                ddos_samples = ddos_df.sample(n=samples_per_type, replace=True, random_state=42)
                selected_samples.append(ddos_samples)
                logger.info(f"Selected {len(ddos_samples)} DDoS samples (oversampled if needed)")
            else:
                logger.warning("Không tìm thấy DDoS samples trong dataset!")

            # 4. Chọn PortScan (5 samples) - Oversample nếu cần
            portscan_df = df[df[label_col].isin(portscan_labels)]
            if len(portscan_df) > 0:
                # Oversample nếu cần để có đúng 5 mẫu
                portscan_samples = portscan_df.sample(n=samples_per_type, replace=True, random_state=42)
                selected_samples.append(portscan_samples)
                logger.info(f"Selected {len(portscan_samples)} PortScan samples (oversampled if needed)")
            else:
                logger.warning("Không tìm thấy PortScan samples trong dataset!")
            
            # Gộp tất cả samples
            if selected_samples:
                samples_df = pd.concat(selected_samples, ignore_index=True)
                # Xáo trộn để không gửi theo thứ tự
                samples_df = samples_df.sample(frac=1, random_state=42).reset_index(drop=True)
            else:
                logger.warning("Không chọn được mẫu nào, sử dụng random selection")
                if len(df) < num_samples:
                    samples_df = df
                else:
                    random_indices = random.sample(range(len(df)), num_samples)
                    samples_df = df.iloc[random_indices]
        
        samples = samples_df.to_dict('records')
        logger.info(f"Selected {len(samples)} samples for attack simulation.")
        
        # Log phân bố các loại tấn công được chọn và mẫu nào chưa có
        if label_col and label_col in samples_df.columns:
            label_counts = samples_df[label_col].value_counts()
            logger.info("=" * 60)
            logger.info("SUMMARY: Attack types distribution in selected samples:")
            logger.info("=" * 60)
            for label, count in label_counts.items():
                logger.info(f"  [OK] {label}: {count} samples")
            
            # Kiểm tra các loại tấn công cần thiết đã có chưa
            logger.info("")
            logger.info("=" * 60)
            logger.info("CHECK: Required attack types availability:")
            logger.info("=" * 60)
            
            # Kiểm tra BENIGN
            benign_found = any(label in benign_labels for label in label_counts.index)
            if benign_found:
                benign_count = sum(count for label, count in label_counts.items() if label in benign_labels)
                logger.info(f"  [OK] BENIGN: {benign_count} samples")
            else:
                logger.warning("  [X] BENIGN: NOT FOUND in selected samples")
            
            # Kiểm tra DoS attacks
            dos_found = any(label in dos_labels for label in label_counts.index)
            if dos_found:
                dos_count = sum(count for label, count in label_counts.items() if label in dos_labels)
                dos_types = [label for label in label_counts.index if label in dos_labels]
                logger.info(f"  [OK] DoS attacks: {dos_count} samples")
                logger.info(f"    Types found: {', '.join(dos_types)}")
            else:
                logger.warning("  [X] DoS attacks: NOT FOUND in selected samples")
            
            # Kiểm tra DDoS
            ddos_found = any(label in ddos_labels for label in label_counts.index)
            if ddos_found:
                ddos_count = sum(count for label, count in label_counts.items() if label in ddos_labels)
                ddos_types = [label for label in label_counts.index if label in ddos_labels]
                logger.info(f"  [OK] DDoS: {ddos_count} samples")
                logger.info(f"    Types found: {', '.join(ddos_types)}")
            else:
                logger.warning("  [X] DDoS: NOT FOUND in selected samples")
            
            # Kiểm tra PortScan
            portscan_found = any(label in portscan_labels for label in label_counts.index)
            if portscan_found:
                portscan_count = sum(count for label, count in label_counts.items() if label in portscan_labels)
                portscan_types = [label for label in label_counts.index if label in portscan_labels]
                logger.info(f"  [OK] PortScan: {portscan_count} samples")
                logger.info(f"    Types found: {', '.join(portscan_types)}")
            else:
                logger.warning("  [X] PortScan: NOT FOUND in selected samples")
            
            logger.info("=" * 60)
            
            # Summary về prediction flow
            logger.info("")
            logger.info("=" * 60)
            logger.info("PREDICTION SUMMARY: Expected prediction flow (CNN):")
            logger.info("=" * 60)
            
            # Mapping từ label sang label_group (theo logic trong preprocess_dataset.py)
            label_to_group = {}
            for label in label_counts.index:
                label_lower = str(label).lower().strip()
                if label_lower in ['benign']:
                    label_to_group[label] = 'benign'
                elif any(dos in label_lower for dos in ['dos hulk', 'dos goldeneye', 'dos slowloris', 'dos slowhttptest', 'dos attack', 'heartbleed']):
                    label_to_group[label] = 'dos'
                elif label_lower in ['ddos', 'ddos attack-hoic', 'ddos attack-loic-udp']:
                    label_to_group[label] = 'ddos'
                elif label_lower in ['portscan']:
                    label_to_group[label] = 'portscan'
                # Bot, Infiltration, Heartbleed đã được bỏ khỏi dataset
                # Không còn rare_attack group
                else:
                    label_to_group[label] = 'other'
            
            # Nhóm theo label_group
            group_counts = {}
            for label, count in label_counts.items():
                group = label_to_group.get(label, 'other')
                if group not in group_counts:
                    group_counts[group] = {}
                group_counts[group][label] = count
            
            # Log prediction flow
            for group in ['benign', 'dos', 'ddos', 'portscan', 'other']:
                if group in group_counts:
                    total = sum(group_counts[group].values())
                    logger.info(f"")
                    logger.info(f"  Level 1 Prediction: '{group}' ({total} samples)")
                    logger.info(f"    → Level 2 Prediction: {'YES' if group == 'dos' else 'NO (not routed to Level 2)'}")
                    if group == 'dos':
                        logger.info(f"    → Level 2 will classify into:")
                        for label, count in group_counts[group].items():
                            logger.info(f"        - {label}: {count} samples")
                    else:
                        logger.info(f"    → Samples:")
                        for label, count in group_counts[group].items():
                            logger.info(f"        - {label}: {count} samples")
            
            logger.info("=" * 60)
            
            # Log tất cả các loại có trong dataset (để tham khảo)
            if label_col in df.columns:
                all_labels = df[label_col].value_counts()
                logger.info("")
                logger.info("=" * 60)
                logger.info("REFERENCE: All attack types available in dataset:")
                logger.info("=" * 60)
                for label, count in all_labels.items():
                    status = "[OK]" if label in label_counts.index else "[  ]"
                    logger.info(f"  {status} {label}: {count} total samples")
                logger.info("=" * 60)

        # RESET OFFSET TRƯỚC KHI GỬI DỮ LIỆU MỚI
        # Điều này đảm bảo cnn_data_preprocessing_service chỉ đọc dữ liệu mới từ lần chạy này,
        # không đọc message cũ từ lần chạy trước
        reset_consumer_group_offset(
            kafka_bootstrap_servers=kafka_bootstrap_servers,
            topic=topic,
            group_id='safenet-cnn-preprocessing-group'
        )

        # Đợi lâu hơn để đảm bảo offset đã được commit và consumer đã rejoin group
        logger.info("Waiting for offset commit and consumer rejoin (5 seconds)...")
        time.sleep(5)

        producer = NetworkDataProducer(kafka_bootstrap_servers=kafka_bootstrap_servers, topic=topic)

        for i, sample in enumerate(samples):
            # Cập nhật timestamp để phản ánh thời điểm gửi hiện tại
            sample['timestamp'] = datetime.now().isoformat()

            # Đảm bảo có ID và Label để đối chiếu
            sample_id = sample.get('id', f"sim_{i}")
            sample_label = sample.get('Label', sample.get('label', 'Unknown'))

            # Gửi dữ liệu
            producer.send_network_data(sample)
            logger.info(f"Sent sample {i+1}/{len(samples)} to Kafka. ID: {sample_id}, LABEL: {sample_label}")
            time.sleep(0.2) # Dừng 0.2 giây giữa các lần gửi để đảm bảo consumer kịp xử lý

        producer.stop()
        logger.info(f"✅ Attack simulation completed successfully! Sent {len(samples)} samples to Kafka topic '{topic}'")
        logger.info(f"📊 Expected distribution: BENIGN:5, DoS:20, DDoS:5, PortScan:5 = Total:35 samples")
        logger.info(f"⏱️  CNN services should now process these samples. Check CNN preprocessing logs for results.")

    except FileNotFoundError:
        logger.error(f"Error: {pkl_file} not found. Please ensure the dataset file exists.")
        logger.info("Note: Default file is 'dataset_clean_cnn.pkl' in the project root directory.")
    except Exception as e:
        logger.error(f"An error occurred during attack simulation: {e}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Simulate Attack for IDS Testing (CNN)')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples to simulate per attack type (and benign)')
    parser.add_argument('--pkl-file', default='dataset.pkl', help='PKL file to load data from')
    parser.add_argument('--kafka-servers', default='localhost:9092', help='Kafka bootstrap servers')
    parser.add_argument('--topic', default='raw_data_event_cnn', help='Kafka topic to send data to')
    parser.add_argument('--force-reset', action='store_true', default=True, help='Force reset consumer offset')

    args = parser.parse_args()

    # Run simulation
    simulate_attack_from_pkl(
        num_samples=args.num_samples,
        pkl_file=args.pkl_file,
        kafka_bootstrap_servers=args.kafka_servers,
        topic=args.topic,
        force_reset=args.force_reset
    )
