# Safenet IDS – Hướng dẫn triển khai

## 1. Giới thiệu
- **Safenet IDS** là hệ thống phát hiện xâm nhập áp dụng mô hình phân cấp (hierarchical classification) trên bộ dữ liệu mạng như CICIDS2017.
- Pipeline tách thành 2 tầng:
  - **Level 1**: Phân loại nhóm tấn công tổng quát (`benign`, `dos`, `ddos`, `bot`, `rare_attack`).
  - **Level 2**: Các mô hình con chuyên biệt cho nhóm `dos` và `rare_attack`, giúp nhận diện chi tiết từng loại.
- Bộ source hiện hỗ trợ cả **triển khai offline** (training, đánh giá) và **mở rộng sang online** thông qua Apache Kafka.

## 2. Cấu trúc thư mục chính
- `scripts/`
  - `preprocess_dataset.py`: tiền xử lý, cân bằng dữ liệu, tạo `label_group`.
  - `split_dataset.py`: chia tập train/val/test (đơn tầng & phân cấp).
  - `load_dataset.py`: tiện ích đọc file dữ liệu.
- `ids_pipeline/`
  - `train_model.py`: huấn luyện mô hình Level 1.
  - `train_model_level2.py`: huấn luyện Level 2 theo từng nhóm.
  - `evaluate_level1.py`, `evaluate_level2.py`: đánh giá và sinh báo cáo.
- `dataset/`: dữ liệu gốc và các file split sinh ra.
- `artifacts/`, `artifacts_level2/`: model, metrics và metadata sau khi huấn luyện.
- `reports/`: báo cáo đánh giá (confusion matrix, ROC, precision/recall).
- `Thiet_ke_trien_khai_IDS.md`: tài liệu thiết kế & checklist triển khai offline.

## 3. Yêu cầu hệ thống
- Python ≥ 3.10.
- Pipenv/venv để tạo môi trường ảo.
- Thư viện chính: `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`, `imbalanced-learn` (tùy chọn), `tensorflow/keras` (nếu cần lưu H5).
- Đối với triển khai online: Apache Kafka (ví dụ `localhost:9092`).

## 4. Thiết lập môi trường
```powershell
cd safenetIDS
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt  # (nếu chưa có, có thể pip install thủ công theo nhu cầu)
```

## 5. Chuẩn bị dữ liệu
- Đặt file gốc (`.csv`/`.pcap`) vào `dataset/` hoặc thư mục con tương ứng.
- Ví dụ CICIDS2017: `dataset/MachineLearningCVE/*.csv`.

## 6. Tiền xử lý dữ liệu
Script hỗ trợ:
- Làm sạch, drop duplicate.
- Xử lý thiếu, convert numeric với `errors="coerce"`.
- IQR clipping (lọc outlier).
- Chuẩn hóa (`standard` hoặc `minmax`).
- One-hot encoding.
- Tạo `label_group` & `label_group_encoded`.
- Cân bằng: oversample/undersample theo nhóm.

Ví dụ:
```powershell
python scripts/preprocess_dataset.py `
  --source dataset/dataset.pkl `
  --output dataset_clean.pkl `
  --output-csv dataset_clean.csv `
  --drop-duplicates `
  --drop-constant-columns `
  --min-non-null-ratio 0.6 `
  --outlier-method iqr_clip `
  --iqr-factor 1.5 `
  --scale-method standard `
  --one-hot `
  --balance-method oversample `
  --summary
```

Metadata tiền xử lý được lưu tại `--metadata-output` (nếu chỉ định).

## 7. Chia dữ liệu (train/val/test)
### 7.1. Level 1 (đơn tầng)
```powershell
python scripts/split_dataset.py `
  --source dataset_clean.pkl `
  --output-dir dataset/splits/level1 `
  --label-column label_group_encoded `
  --train-size 0.7 `
  --val-size 0.15 `
  --test-size 0.15 `
  --stratify `
  --balance-train oversample
```

### 7.2. Level 2 (phân cấp)
```powershell
python scripts/split_dataset.py `
  --source dataset_clean.pkl `
  --hierarchical `
  --group-column label_group `
  --level2-groups dos rare_attack `
  --output-dir dataset/splits/level2
```

Script sẽ tạo các thư mục con `level1/`, `level2/dos/`, `level2/rare_attack/` chứa `train_raw.pkl`, `train_balanced.pkl`, `val.pkl`, `test.pkl` và `split_summary.json`.

## 8. Huấn luyện
### 8.1. Level 1
```powershell
python ids_pipeline/train_model.py `
  --splits-dir dataset/splits/level1 `
  --train-variant balanced `
  --output-dir artifacts `
  --auto-split  # tùy chọn, tự chạy split nếu thiếu file
```
Kết quả: `ids_pipeline.joblib`, (tùy chọn) `ids_pipeline_model.h5`, `metrics.json`, `metadata.json`.

### 8.2. Level 2
```powershell
python ids_pipeline/train_model_level2.py `
  --groups dos rare_attack `
  --splits-dir dataset/splits/level2 `
  --output-dir artifacts_level2 `
  --train-variant balanced
```
Sinh mô hình cho từng nhóm: `artifacts_level2/dos/` và `artifacts_level2/rare_attack/`.

## 9. Đánh giá & báo cáo
### 9.1. Level 1
```powershell
python ids_pipeline/evaluate_level1.py `
  --splits-dir dataset/splits/level1 `
  --model-path artifacts/ids_pipeline.joblib `
  --output-dir reports/level1_eval
```
Sinh: `metrics.json`, `per_class_metrics.csv`, `confusion_matrix.png`, `prf_per_class.png`, biểu đồ ROC/PR.

### 9.2. Level 2
```powershell
python ids_pipeline/evaluate_level2.py `
  --groups dos rare_attack `
  --splits-root dataset/splits/level2 `
  --models-root artifacts_level2 `
  --output-root reports/level2_eval
```
Mỗi nhóm có báo cáo riêng trong `reports/level2_eval/<group>/`.

## 10. Kiến trúc triển khai online (mở rộng)
- **Kafka topics**:
  - `raw_network_events`, `preprocessed_events`, `level1_predictions`, `level2_predictions`, `ids_alerts`.
- **Các service**:
  - Producer thu thập dữ liệu (PCAP/log) ⇒ `raw_network_events`.
  - Consumer tiền xử lý (tái sử dụng logic từ `preprocess_dataset.py`) ⇒ `preprocessed_events`.
  - Consumer Level 1 (load `ids_pipeline.joblib`) ⇒ `level1_predictions`.
  - Consumer Level 2 (mỗi nhóm một service, load model tương ứng) ⇒ `level2_predictions`.
  - Alerting & storage: ghi vào DB, Dashboard (Grafana/Streamlit).
- Chi tiết thêm tại `Thiet_ke_trien_khai_IDS.md`.

## 11. Ghi chú
- **Log & metadata**: luôn kiểm tra `reports/` và `artifacts/` để xem thông số training.
- **Cân bằng dữ liệu**: có thể chuyển từ oversample sang undersample (`--balance-method undersample`) tùy bài toán.
- **Nâng cấp**:
  - Bổ sung PCA/feature selection (hiện chưa tích hợp mặc định).
  - Thêm lớp model khác (XGBoost, LightGBM, Deep Learning).
  - Viết service Kafka producer/consumer cụ thể cho online inference.

## 12. Liên hệ & đóng góp
- Issues/feature requests: tạo ticket trên repository hoặc mở discussion.
- Khi thêm chức năng mới, đảm bảo cập nhật README và `Thiet_ke_trien_khai_IDS.md`.


