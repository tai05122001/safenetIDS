# Thiết kế triển khai Intrusion Detection System

## 1. Tổng quan

Hệ thống gồm **6 module chính**, tương ứng pipeline chi tiết:

1.  **Network Traffic Collection Module**
2.  **Flow Generation (Zeek/Bro)**
3.  **Preprocessing Module (Data Cleaning & Encoding)**
4.  **Feature Extraction Module (CICFlowMeter / Custom Extractor)**
5.  **ML Model Module (Training & Inference)**
6.  **Alerting & Visualization Module (Kafka + Dashboard)**

> *Hình 1: Pipeline Intrusion Detection System*

------------------------------------------------------------------------

## 2. Network Traffic Collection Module

**Mục tiêu:**\
Thu thập dữ liệu mạng để làm đầu vào cho hệ thống.

**Giải thích:**\
Giống như việc "ghi lại" mọi gói tin đi qua mạng, giúp hệ thống có dữ
liệu để phát hiện hành vi bất thường.

**Cách triển khai:**

-   **Offline:**\
    Dùng sẵn dữ liệu mẫu có sẵn (ví dụ bộ *CICIDS2017*). Dữ liệu này gồm
    nhiều dạng tấn công thật (DDoS, Botnet, PortScan, v.v.).

-   **Online:**\
    Nếu có thiết bị mạng thật, có thể dùng công cụ **Zeek** hoặc
    **Wireshark** để "nghe" trực tiếp các gói tin đang truyền qua.

**Kết quả:**\
Sinh ra các tệp dữ liệu dạng `.pcap` hoặc `log`, chứa thông tin kết nối
giữa các máy (ai nói chuyện với ai, bao nhiêu byte, giao thức gì,...).

------------------------------------------------------------------------

## 3. Flow Generation (Zeek/Bro)

**Mục tiêu:**\
Biến dữ liệu mạng thô thành dạng có cấu trúc -- dễ phân tích bằng máy
học.

**Giải thích:**\
Thay vì xem từng gói tin (rất nhiều và rời rạc), ta gom các gói tin
thành một "luồng" (flow) -- tức là một cuộc trao đổi hoàn chỉnh giữa hai
máy (ví dụ: A gửi dữ liệu đến B trong 5 giây).

**Cách triển khai:**

-   Dùng phần mềm **Zeek (trước đây là Bro)** để đọc file `.pcap` và tạo
    ra file log chi tiết (ví dụ: `conn.log`, `http.log`, `dns.log`,
    ...).
-   Mỗi dòng log biểu diễn một kết nối (flow) với các thông tin cơ bản:
    -   Địa chỉ IP nguồn & đích\
    -   Thời lượng kết nối\
    -   Số lượng byte và gói tin gửi đi -- nhận về\
    -   Giao thức (TCP, UDP, HTTP, ...)

**Kết quả:**\
File log ở dạng CSV -- mỗi dòng là một flow, sẽ được xử lý ở bước kế
tiếp.

------------------------------------------------------------------------

## 4. Preprocessing Module (Data Cleaning & Encoding)

**Mục tiêu:**\
Làm sạch dữ liệu, chuyển về dạng mà mô hình có thể hiểu và học được.

**Giải thích:**\
Dữ liệu gốc có thể chứa lỗi, giá trị trống, hoặc ký hiệu chữ (ví dụ
"TCP", "UDP"), mà mô hình chỉ hiểu số.\
Ta cần chuẩn hóa và mã hóa dữ liệu này.

**Cách triển khai:**

-   Loại bỏ dòng lỗi hoặc trống\
-   Thay thế giá trị trống bằng số 0 hoặc trung bình\
-   Biến đổi ký hiệu chữ thành số (VD: `TCP → 1`, `UDP → 2`)\
-   Chuẩn hóa dữ liệu về cùng thang đo (để các đặc trưng như "số byte"
    hay "thời lượng" không chênh lệch quá lớn)

**Kết quả:**\
Một bảng dữ liệu sạch, toàn số -- sẵn sàng để mô hình học.

------------------------------------------------------------------------

## 5. Feature Extraction Module (CICFlowMeter / Custom Extractor)

**Mục tiêu:**\
Tạo ra các đặc trưng (feature) giúp mô hình phân biệt bình thường và bất
thường.

**Giải thích:**\
Các "feature" giống như các đặc điểm của một hành vi mạng. Ví dụ:

-   Một người gửi hàng nghìn yêu cầu trong 1 giây → có thể là tấn công
    DDoS.\
-   Một IP kết nối với hàng trăm địa chỉ khác → có thể là Botnet.

**Cách triển khai:**

-   Dùng công cụ **CICFlowMeter** hoặc **script Python** để tạo ra các
    đặc trưng như:
    -   Thời lượng kết nối\
    -   Số gói tin gửi/nhận\
    -   Tổng byte gửi/nhận\
    -   Độ trễ giữa các gói tin\
    -   Tỷ lệ byte chiều đi / chiều về\
-   Có thể lọc giữ lại các đặc trưng quan trọng nhất để mô hình học hiệu
    quả hơn.

**Kết quả:**\
Một bảng dữ liệu (nhiều cột -- mỗi cột là 1 đặc trưng), dùng làm đầu vào
huấn luyện mô hình.

------------------------------------------------------------------------

## 6. ML Model Module (Training & Inference)

**Mục tiêu:**\
Dạy máy tính phát hiện hành vi bất thường trong mạng.

**Giải thích:**\
Máy tính sẽ "học" từ dữ liệu mạng bình thường và tấn công, sau đó có thể
tự nhận ra khi có hành vi khác thường.

**Cách triển khai:**

### Khi **không có nhãn (unsupervised)**:

-   Dùng mô hình như **Isolation Forest** hoặc **VAE (Variational
    Autoencoder)** để phát hiện các dòng dữ liệu khác biệt với phần lớn
    còn lại.

### Khi **có nhãn (supervised)**:

-   Dùng mô hình như **Random Forest**, **XGBoost**, hoặc **MLP** để
    phân loại theo loại tấn công.

**Đánh giá kết quả:** - Dùng các chỉ số như: - Precision\
- Recall\
- F1-score\
- False Positive Rate (tỷ lệ cảnh báo sai)

**Kết quả:**\
Một mô hình được huấn luyện sẵn, có thể đọc luồng dữ liệu mới và gắn
nhãn "bình thường" hoặc "bị tấn công".

------------------------------------------------------------------------

## 7. Alerting & Visualization Module (Kafka + Dashboard)

**Mục tiêu:**\
Hiển thị cảnh báo và lưu kết quả phát hiện bất thường.

**Giải thích:**\
Khi mô hình phát hiện điều bất thường, hệ thống cần hiển thị ngay cho
người quản trị mạng hoặc lưu vào file log.

**Cách triển khai:**

-   Dùng **Kafka** để truyền dữ liệu thời gian thực từ mô hình đến bảng
    điều khiển (Dashboard).
-   **Dashboard** (làm bằng *Streamlit* hoặc *Flask*) hiển thị thông tin
    như:
    -   Địa chỉ IP bị nghi ngờ\
    -   Loại tấn công\
    -   Thời điểm xảy ra\
    -   Mức độ nghiêm trọng\
-   Có thể xuất báo cáo định kỳ hoặc lưu vào cơ sở dữ liệu để phân tích
    sau.

**Kết quả:**\
Một giao diện trực quan giúp quản trị viên xem cảnh báo và tra cứu chi
tiết từng sự cố.

------------------------------------------------------------------------

## 8. Checklist triển khai **offline**

**Mục tiêu:**\
Đảm bảo có thể tái lập toàn bộ pipeline IDS với dữ liệu mẫu (không cần mạng thật).

**Bước thực hiện:**

-   **Chuẩn bị dữ liệu gốc**
    -   Tải bộ dữ liệu `.pcap` hoặc `.csv` (ví dụ `CICIDS2017`) và đặt tại `data/raw/`.
    -   (Tuỳ chọn) Dùng Zeek để chuyển `.pcap` → `.log`/`.csv` nếu cần chuẩn hóa cấu trúc.

-   **Tiền xử lý & trích đặc trưng**
    -   Sử dụng script `scripts/preprocess_dataset.py` với các tuỳ chọn mới để làm sạch, loại bỏ dòng/cột lỗi, one-hot encoding và chuẩn hóa số liệu.
    -   Ví dụ:

        ```
        python scripts/preprocess_dataset.py \
          --source data/raw/cicids2017.csv \
          --output data/processed/cicids2017_clean.pkl \
          --output-csv data/processed/cicids2017_clean.csv \
          --metadata-output artifacts/cicids2017_preprocess.json \
          --drop-duplicates \
          --drop-constant-columns \
          --min-non-null-ratio 0.6 \
          --outlier-method iqr_clip \
          --iqr-factor 1.5 \
          --scale-method standard \
          --one-hot \
          --summary
        ```

    -   File `artifacts/cicids2017_preprocess.json` lưu metadata (mapping nhãn, cột đã loại bỏ, thống kê scaling) phục vụ bước huấn luyện/inference sau này.

-   **Huấn luyện mô hình**
    -   Dùng dữ liệu từ `data/processed/` để huấn luyện mô hình ML đã chọn.
    -   Lưu checkpoint (`.pkl`, `.onnx`, ...) cùng cấu hình huấn luyện để tái sử dụng.

-   **Đánh giá & báo cáo**
    -   Sinh các chỉ số: Precision, Recall, F1, False Positive Rate.
    -   Ghi nhận ma trận nhầm lẫn, biểu đồ ROC/PR và lưu vào thư mục `reports/`.

-   **Tài liệu vận hành**
    -   Ghi lại toàn bộ lệnh chạy, yêu cầu môi trường (phiên bản Python/thư viện), cấu trúc thư mục dữ liệu.
    -   Bổ sung kết quả chạy thử (log, báo cáo) để người tích hợp có thể kiểm chứng nhanh.

**Kết quả:**\
Một quy trình offline hoàn chỉnh, dữ liệu sạch chuẩn hoá, mô hình đã kiểm thử cùng hướng dẫn tái lập.
