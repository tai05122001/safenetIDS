import pandas as pd
from services.network_data_producer import NetworkDataProducer
import time
import random
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('services/logs/simulate_attack.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SimulateAttack')

def simulate_attack_from_pkl(num_samples=20, pkl_file='dataset.pkl', kafka_bootstrap_servers='localhost:9092', topic='raw_data_event'):
    """
    Tải dữ liệu từ dataset.pkl, chọn ngẫu nhiên các mẫu và gửi chúng đến Kafka.
    """
    try:
        df = pd.read_pickle(pkl_file)
        logger.info(f"Loaded {len(df)} records from {pkl_file}")

        # Lấy 20 mẫu ngẫu nhiên
        if len(df) < num_samples:
            logger.warning(f"Dataset has only {len(df)} samples, less than requested {num_samples}. Sending all available samples.")
            samples_df = df
        else:
            random_indices = random.sample(range(len(df)), num_samples)
            samples_df = df.iloc[random_indices]
        
        samples = samples_df.to_dict('records')
        logger.info(f"Selected {len(samples)} samples for attack simulation.")

        producer = NetworkDataProducer(kafka_bootstrap_servers=kafka_bootstrap_servers, topic=topic)

        for i, sample in enumerate(samples):
            # Cập nhật timestamp để phản ánh thời điểm gửi hiện tại
            sample['timestamp'] = datetime.now().isoformat()
            
            # Gửi dữ liệu
            producer.send_network_data(sample)
            logger.info(f"Sent sample {i+1}/{len(samples)} to Kafka.")
            time.sleep(0.1) # Dừng một chút giữa các lần gửi

        producer.stop()
        logger.info("Attack simulation completed.")

    except FileNotFoundError:
        logger.error(f"Error: {pkl_file} not found. Please ensure the dataset file exists.")
    except Exception as e:
        logger.error(f"An error occurred during attack simulation: {e}")

if __name__ == '__main__':
    # Tạo thư mục logs nếu chưa tồn tại
    import os
    os.makedirs('services/logs', exist_ok=True)
    
    # Run simulation
    simulate_attack_from_pkl(num_samples=20, pkl_file='dataset.pkl')
