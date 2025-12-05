# Thiết kế triển khai Intrusion Detection System

## 1. Tổng quan

Hệ thống Safenet IDS là một hệ thống phát hiện xâm nhập đa cấp sử dụng kiến trúc microservices với Apache Kafka, bao gồm **8 module chính**:

1.  **Network Traffic Collection Module** (Packet Capture Service)
2.  **Data Preprocessing Module** (Kafka-based Service)
3.  **Level 1 Prediction Module** (Random Forest - Phân loại nhóm tấn công)
4.  **Level 2 Prediction Module** (Random Forest - Phân loại chi tiết loại tấn công)
5.  **Level 3 Prediction Module** (Random Forest - Phân loại DoS variants)
6.  **Alerting Module** (Kafka + SQLite Database)
7.  **Attack Simulation Module** (Giả lập tấn công để test)
8.  **Visualization & Monitoring Module** (Dashboard + Logging)

**Kiến trúc tổng thể:**
```
Packet Capture → Kafka → Data Preprocessing → Level 1 Prediction → Level 2 Prediction → Level 3 Prediction → Alerting → Database
```

**Công nghệ sử dụng:**
- **Packet Capture**: pyshark, scapy
- **Message Queue**: Apache Kafka
- **Machine Learning**: scikit-learn, Random Forest
- **Database**: SQLite
- **Visualization**: Streamlit (tương lai)

> *Hình 1: Pipeline Safenet IDS với Kafka Microservices*

------------------------------------------------------------------------

## 2. Network Traffic Collection Module

**Mục tiêu:**\
Thu thập dữ liệu mạng real-time hoặc từ file để làm đầu vào cho hệ thống phát hiện xâm nhập.

**Giải thích:**\
Module này thu thập các gói tin mạng và chuyển đổi thành dữ liệu có cấu trúc để xử lý tiếp theo trong pipeline Kafka.

**Cách triển khai:**

-   **Real-time Capture:**\
    Sử dụng `services/packet_capture_service.py` với pyshark để capture gói tin từ network interface.

-   **File-based Input:**\
    Sử dụng `services/network_data_producer.py` để đọc dữ liệu từ file CSV/PCAP có sẵn (như CICIDS2017).

-   **Simulation Mode:**\
    Sử dụng `services/simulate_attack_service.py` để tạo dữ liệu tấn công giả lập phục vụ testing.

**Công nghệ sử dụng:**
- **pyshark**: Python wrapper cho Wireshark/tshark
- **scapy**: Network packet manipulation (tùy chọn)
- **Kafka Producer**: Gửi dữ liệu đến topic `raw_network_events`

**Kết quả:**\
Dữ liệu được gửi đến Kafka topic `raw_network_events` dưới dạng JSON messages, sẵn sàng cho module preprocessing.

------------------------------------------------------------------------

## 3. Data Preprocessing Module

**Mục tiêu:**\
Làm sạch và chuẩn hóa dữ liệu mạng từ Kafka để chuẩn bị cho việc prediction.

**Giải thích:**\
Module này nhận dữ liệu thô từ Kafka topic `raw_network_events`, thực hiện các bước preprocessing và gửi kết quả đến topic `preprocessed_events`.

**Cách triển khai:**
Sử dụng `services/data_preprocessing_service.py` với các tính năng:

-   **Kafka Consumer/Producer:** Đọc từ `raw_network_events`, ghi ra `preprocessed_events`
-   **Data Cleaning:**
    -   Chuẩn hóa tên cột
    -   Xử lý missing values (fill bằng 0 hoặc mean)
    -   Loại bỏ outliers bằng IQR method
-   **Feature Engineering:**
    -   Standard scaling cho các đặc trưng numeric
    -   Tạo label_group cho classification
-   **Error Handling:** Retry mechanism và logging chi tiết

**Công nghệ sử dụng:**
- **pandas/numpy**: Data manipulation
- **scikit-learn**: Scaling và preprocessing
- **Kafka**: Message passing

**Kết quả:**\
Dữ liệu đã được làm sạch và chuẩn hóa, sẵn sàng cho các module prediction tiếp theo.

------------------------------------------------------------------------

## 4. Level 1 Prediction Module (Binary Classification)

**Mục tiêu:**\
Phân loại traffic là bình thường hay có dấu hiệu tấn công.

**Giải thích:**\
Đây là cấp độ đầu tiên của hệ thống 3-level prediction, sử dụng Random Forest để phân loại cơ bản giữa traffic benign và malicious.

**Cách triển khai:**
Sử dụng `services/random_forest/level1_prediction_service_rf.py`:

-   **Kafka Integration:** Consumer `preprocessed_events` → Producer `level1_predictions`
-   **Model:** Random Forest đã được train từ `ids_pipeline/train_level1_rf.py`
-   **Classification:** Phân loại thành các nhóm: benign, dos, ddos, portscan
-   **Output:** Prediction kết quả kèm confidence score

**Model Training:**
```bash
python ids_pipeline/train_level1_rf.py
```

**Kết quả:**\
Traffic được gắn nhãn cấp độ 1, tiếp tục được xử lý bởi Level 2 nếu phát hiện tấn công.

------------------------------------------------------------------------

## 5. Level 2 Prediction Module (Attack Type Classification)

**Mục tiêu:**\
Phân loại chi tiết loại tấn công dựa trên kết quả từ Level 1.

**Giải thích:**\
Khi Level 1 phát hiện traffic là malicious, Level 2 sẽ phân loại chi tiết hơn về loại tấn công cụ thể.

**Cách triển khai:**
Sử dụng `services/random_forest/level2_prediction_service_rf.py`:

-   **Conditional Processing:** Chỉ kích hoạt khi Level 1 detect malicious traffic
-   **Kafka Integration:** Consumer `level1_predictions` → Producer `level2_predictions`
-   **Models:** Nhiều model Random Forest cho từng loại tấn công (DoS, DDoS, PortScan)
-   **Attack Types:** DoS Hulk, DoS GoldenEye, DoS Slowloris, DoS Slowhttptest, v.v.

**Model Training:**
```bash
python ids_pipeline/train_level2_attack_types_rf.py
# hoặc
python ids_pipeline/train_level2_rf.py
```

**Kết quả:**\
Traffic malicious được phân loại chi tiết về loại tấn công, chuyển tiếp cho Level 3 nếu cần.

------------------------------------------------------------------------

## 6. Level 3 Prediction Module (DoS Attack Variants)

**Mục tiêu:**\
Phân loại chi tiết các biến thể tấn công DoS.

**Giải thích:**\
Khi Level 2 xác định là tấn công DoS, Level 3 sẽ phân loại sâu hơn về các biến thể cụ thể của DoS attacks.

**Cách triển khai:**
Sử dụng `services/random_forest/level3_prediction_service_rf.py`:

-   **Conditional Processing:** Chỉ kích hoạt khi Level 2 detect DoS attacks
-   **Kafka Integration:** Consumer `level2_predictions` → Producer `level3_predictions`
-   **Model:** Random Forest chuyên biệt cho DoS classification
-   **DoS Variants:** Hulk, GoldenEye, Slowloris, Slowhttptest, v.v.

**Model Training:**
```bash
python ids_pipeline/train_level3_dos_rf.py
```

**Kết quả:**\
Tấn công DoS được phân loại chi tiết về biến thể cụ thể.

------------------------------------------------------------------------

## 7. Alerting Module (Kafka + SQLite Database)

**Mục tiêu:**\
Tạo và quản lý cảnh báo bảo mật từ kết quả prediction.

**Giải thích:**\
Module này nhận kết quả từ Level 3 predictions, tạo alerts dựa trên confidence thresholds và lưu vào database.

**Cách triển khai:**
Sử dụng `services/alerting_service.py`:

-   **Kafka Integration:** Consumer `level3_predictions` → Producer `ids_alerts`
-   **Alert Generation:**
    -   Threshold-based: Chỉ tạo alert khi confidence > ngưỡng
    -   Severity Classification: low/medium/high/critical
    -   Alert Types: DoS, DDoS, PortScan, etc.
-   **Database Storage:** SQLite database tại `services/data/alerts.db`
-   **Alert Schema:** timestamp, source_ip, attack_type, confidence, severity

**Alert Thresholds:**
```python
alert_thresholds = {
    'benign': 0.0,      # Không tạo alert
    'dos': 0.7,         # 70% confidence
    'ddos': 0.6,        # 60% confidence
    'portscan': 0.65,   # 65% confidence
}
```

**Kết quả:**\
Cảnh báo được lưu trữ và có thể truy vấn để monitoring và response.

------------------------------------------------------------------------

## 8. Attack Simulation Module

**Mục tiêu:**\
Giả lập các cuộc tấn công mạng để kiểm thử và đánh giá hệ thống IDS.

**Giải thích:**\
Module này tạo ra traffic tấn công giả lập để test toàn bộ pipeline từ capture đến alerting.

**Cách triển khai:**
Sử dụng `services/simulate_attack_service.py`:

-   **Attack Types:** DoS, DDoS, PortScan, Brute Force
-   **Traffic Generation:** Tạo packets với các pattern tấn công thực tế
-   **Integration:** Gửi dữ liệu trực tiếp vào pipeline (bỏ qua packet capture)
-   **Testing Scenarios:** Load testing, accuracy validation

**Cách sử dụng:**
```bash
python services/simulate_attack_service.py --attack-type dos --duration 60
```

**Kết quả:**\
Dữ liệu tấn công giả lập để validate hệ thống phát hiện.

------------------------------------------------------------------------

## 9. Visualization & Monitoring Module

**Mục tiêu:**\
Cung cấp giao diện giám sát và báo cáo cho hệ thống IDS.

**Giải thích:**\
Module này hiển thị alerts, metrics và logs của hệ thống trong thời gian thực.

**Cách triển khai (tương lai):**
-   **Dashboard:** Streamlit web interface
-   **Real-time Monitoring:** Kafka consumer để hiển thị alerts
-   **Historical Reports:** Query từ SQLite database
-   **Metrics Visualization:** Charts cho accuracy, throughput, false positives

**Tính năng dự kiến:**
-   Live alerts feed
-   Attack statistics
-   System performance metrics
-   Log viewer
-   Alert management interface

------------------------------------------------------------------------

## 10. Checklist triển khai **offline**

**Mục tiêu:**\
Triển khai hoàn chỉnh hệ thống IDS với dữ liệu mẫu, bao gồm training pipeline và inference services.

**Bước thực hiện:**

### 1. Chuẩn bị môi trường
```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Setup Kafka (nếu chưa có)
# Download và cài đặt Kafka từ kafka.apache.org
# Chạy Zookeeper và Kafka server
```

### 2. Chuẩn bị dữ liệu training
```bash
# Download CICIDS2017 dataset và đặt vào thư mục phù hợp
# Sử dụng extract_samples.py để trích xuất samples nếu cần

# Load và kiểm tra dataset
python scripts/load_dataset.py

# Preprocess dataset
python scripts/preprocess_dataset.py

# Chia tập train/test
python scripts/split_dataset.py
```

### 3. Training Models (3 Level Pipeline)
```bash
# Level 1: Binary classification (benign vs attack groups)
python ids_pipeline/train_level1_rf.py

# Level 2: Attack type classification
python ids_pipeline/train_level2_attack_types_rf.py

# Level 3: DoS attack variants
python ids_pipeline/train_level3_dos_rf.py
```

### 4. Evaluate Models
```bash
# Đánh giá Level 1
python ids_pipeline/evaluate_level1.py

# Đánh giá Level 2
python ids_pipeline/evaluate_level2.py
```

### 5. Khởi động Inference Services
```bash
# Cách 1: Khởi động tất cả services
cd services
start_all_services.bat

# Cách 2: Khởi động từng service
# Terminal 1: Network Data Producer
python services/network_data_producer.py

# Terminal 2: Data Preprocessing
python services/data_preprocessing_service.py

# Terminal 3: Level 1 Prediction
python services/random_forest/level1_prediction_service_rf.py

# Terminal 4: Level 2 Prediction
python services/random_forest/level2_prediction_service_rf.py

# Terminal 5: Level 3 Prediction
python services/random_forest/level3_prediction_service_rf.py

# Terminal 6: Alerting Service
python services/alerting_service.py
```

### 6. Test hệ thống
```bash
# Test với dữ liệu giả lập
python services/simulate_attack_service.py --attack-type dos --duration 30

# Kiểm tra alerts trong database
sqlite3 services/data/alerts.db "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 5;"
```

### 7. Monitoring và Logs
- **Kafka Topics:** Kiểm tra messages trong các topics: `raw_network_events`, `preprocessed_events`, `level1_predictions`, `level2_predictions`, `level3_predictions`, `ids_alerts`
- **Logs:** Xem logs trong `services/logs/` để debug
- **Database:** Query SQLite database để xem alerts

**Kết quả:**\
Hệ thống IDS hoàn chỉnh với pipeline training và inference services, sẵn sàng phát hiện tấn công mạng thời gian thực.
