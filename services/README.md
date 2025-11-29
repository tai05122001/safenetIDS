# Safenet IDS - Kafka Services

Hướng dẫn triển khai và sử dụng các Kafka services cho hệ thống phát hiện xâm nhập Safenet IDS.

## 📚 Documentation Features

Tất cả code trong thư mục `services/` đã được comment chi tiết bằng tiếng Việt với:
- **Function/Class purposes**: Mô tả rõ ràng chức năng và trách nhiệm
- **Parameter explanations**: Giải thích từng tham số đầu vào
- **Process flow**: Luồng xử lý từng bước một cách chi tiết
- **Error handling**: Cách xử lý exceptions và edge cases
- **Integration points**: Cách service tương tác với Kafka và các components khác

## 🎯 Quick Reference

| File | Service | Input Topic | Output Topic | Purpose |
|------|---------|-------------|--------------|---------|
| `network_data_producer.py` | Network Producer | - | `raw_network_events` | Generate network traffic data |
| `data_preprocessing_service.py` | Data Preprocessing | `raw_network_events` | `preprocessed_events` | Clean & normalize data |
| `level1_prediction_service.py` | Level 1 Prediction | `preprocessed_events` | `level1_predictions` | Classify attack groups |
| `level2_prediction_service.py` | Level 2 Prediction | `level1_predictions` | `level2_predictions` | Detailed attack classification |
| `alerting_service.py` | Alerting Service | `level2_predictions` | `ids_alerts` | Generate security alerts |

## Tổng quan kiến trúc

```
Network Data → raw_network_events → Data Preprocessing → preprocessed_events → Level 1 Prediction → level1_predictions → Level 2 Prediction → level2_predictions → Alerting → ids_alerts
```

## Các Services

### 1. Network Data Producer Service (`network_data_producer.py`)
**Chức năng**: Thu thập dữ liệu network và gửi đến Kafka
- **Input**: Không có (tự tạo sample data hoặc đọc từ file)
- **Output**: `raw_network_events`
- **Cách chạy**:
  ```bash
  python services/network_data_producer.py
  ```
- **Tùy chọn**:
  - `--historical-data`: Đọc dữ liệu từ file CSV để test
  - `--interval`: Khoảng thời gian gửi dữ liệu (giây)

### 2. Data Preprocessing Service (`data_preprocessing_service.py`)
**Chức năng**: Tiền xử lý dữ liệu network theo pipeline của dự án
- **Input**: `raw_network_events`
- **Output**: `preprocessed_events`
- **Cách chạy**:
  ```bash
  python services/data_preprocessing_service.py
  ```
- **Tính năng**:
  - Chuẩn hóa tên cột
  - Convert numeric và fill missing values
  - IQR outlier clipping
  - Standard scaling
  - Tạo label_group

### 3. Level 1 Prediction Service (`level1_prediction_service.py`)
**Chức năng**: Phân loại nhóm attack tổng quát (benign/dos/ddos/portscan)
- **Input**: `preprocessed_events`
- **Output**: `level1_predictions`
- **Model**: `artifacts/ids_pipeline.joblib`
- **Cách chạy**:
  ```bash
  python services/level1_prediction_service.py
  ```
- **Tính năng**:
  - Load model Level 1
  - Chạy prediction với confidence scores
  - Gửi kết quả kèm thông tin model

### 4. Level 2 Prediction Service (`level2_prediction_service.py`)
**Chức năng**: Phân loại chi tiết cho nhóm DoS (dos)
- **Input**: `level1_predictions`
- **Output**: `level2_predictions`
- **Models**: `artifacts_level2/{group}/{group}_pipeline.joblib`
- **Cách chạy**:
  ```bash
  python services/level2_prediction_service.py
  ```
- **Tính năng**:
  - Chỉ chạy Level 2 khi Level 1 detect dos
  - Load model tương ứng theo group
  - Mapping prediction sang attack types cụ thể (DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest)

### 5. Alerting Service (`alerting_service.py`)
**Chức năng**: Tạo và quản lý alerts từ predictions
- **Input**: `level2_predictions`
- **Output**: `ids_alerts`
- **Database**: `services/data/alerts.db`
- **Cách chạy**:
  ```bash
  python services/alerting_service.py
  ```
- **Tính năng**:
  - Tạo alerts dựa trên confidence thresholds
  - Phân loại severity (low/medium/high/critical)
  - Lưu alerts vào SQLite database
  - Gửi alerts đến Kafka topic

## Khởi động hệ thống

### Cách 1: Khởi động tất cả services cùng lúc
```bash
cd services
start_all_services.bat
```

### Cách 2: Khởi động từng service riêng lẻ
```bash
# Terminal 1 - Kafka services
cd c:/kafka
start-ids-kafka.bat

# Terminal 2 - Network Producer
cd services
python network_data_producer.py

# Terminal 3 - Data Preprocessing
python data_preprocessing_service.py

# Terminal 4 - Level 1 Prediction
python level1_prediction_service.py

# Terminal 5 - Level 2 Prediction
python level2_prediction_service.py

# Terminal 6 - Alerting
python alerting_service.py
```

## Cấu hình

### Biến môi trường và tham số mặc định:
- **Kafka Servers**: `localhost:9092`
- **Model Paths**:
  - Level 1: `artifacts/ids_pipeline.joblib`
  - Level 2: `artifacts_level2/{group}/{group}_pipeline.joblib`
- **Database**: `services/data/alerts.db`
- **Logs**: `services/logs/`

### Thay đổi cấu hình:
```bash
# Sử dụng Kafka servers khác
python network_data_producer.py --kafka-servers kafka-cluster:9092

# Thay đổi model path
python level1_prediction_service.py --model-path custom_model.joblib

# Thay đổi database path
python alerting_service.py --db-path custom_alerts.db
```

## Monitoring và Debug

### Logs
Tất cả logs được lưu trong `services/logs/`:
- `network_producer.log`
- `data_preprocessing.log`
- `level1_prediction.log`
- `level2_prediction.log`
- `alerting.log`

### Kiểm tra hoạt động Kafka
```bash
# Kiểm tra topics
bin\windows\kafka-topics.bat --list --bootstrap-server localhost:9092

# Monitor messages
bin\windows\kafka-console-consumer.bat --topic ids_alerts --from-beginning --bootstrap-server localhost:9092
```

### Kiểm tra database alerts
```bash
# Sử dụng SQLite browser hoặc command line
sqlite3 services/data/alerts.db "SELECT * FROM alerts LIMIT 10;"
```

## Alert Thresholds

Cấu hình ngưỡng confidence để tạo alert (trong `alerting_service.py`):

```python
alert_thresholds = {
    'benign': 0.0,      # Không tạo alert
    'dos': 0.7,         # 70% confidence
    'ddos': 0.6,        # 60% confidence
    'portscan': 0.65,   # 65% confidence
    'default': 0.7      # Mặc định 70%
}
```

## Troubleshooting

### Service không kết nối được Kafka
- Kiểm tra Kafka đang chạy: `start-ids-kafka.bat`
- Kiểm tra ports: `netstat -an | find "9092"`
- Kiểm tra logs Kafka trong `c:/kafka/logs/`

### Model không load được
- Kiểm tra file model tồn tại: `dir artifacts\`
- Kiểm tra dependencies: `pip install -r requirements.txt`
- Kiểm tra logs service tương ứng

### Không có alerts được tạo
- Kiểm tra confidence scores trong logs
- Kiểm tra thresholds trong alerting service
- Kiểm tra database: `sqlite3 services/data/alerts.db "SELECT COUNT(*) FROM alerts;"`

## Performance Tuning

### Kafka Configuration
- Tăng `num.partitions` trong `server.properties` cho throughput cao hơn
- Điều chỉnh `batch.size` và `linger.ms` trong producer properties

### Service Configuration
- Tăng `max_poll_records` trong consumer để xử lý nhiều messages cùng lúc
- Điều chỉnh `buffer_memory` trong producer cho memory usage

### Database Optimization
- Thêm indexes cho các trường thường query
- Implement cleanup policy cho alerts cũ

## Mở rộng

### Thêm Data Sources
Sửa `network_data_producer.py` để:
- Đọc từ PCAP files
- Kết nối network interfaces
- Tích hợp với SIEM systems

### Custom Alert Actions
Sửa `alerting_service.py` để:
- Gửi email/SMS alerts
- Tích hợp với ticketing systems
- Trigger automated responses

### Dashboard Integration
- Kết nối với Grafana để visualize alerts
- Tích hợp với Elasticsearch/Kibana stack
- Real-time monitoring dashboard
