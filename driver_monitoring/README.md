# Hệ thống Giám sát An Toàn Giao Thông Bằng AI

Hệ thống giám sát tài xế theo thời gian thực sử dụng trí tuệ nhân tạo để phát hiện các hành vi nguy hiểm như buồn ngủ và mất tập trung.

## Tính năng chính

- **Phát hiện buồn ngủ**:
  - Phát hiện chớp mắt kéo dài
  - Phát hiện ngáp
  - Phát hiện gật gù đầu
  - Sử dụng CNN-LSTM để phân tích chuỗi thời gian

- **Phát hiện mất tập trung**:
  - Phát hiện nhìn trái/phải
  - Phát hiện sử dụng điện thoại
  - Phát hiện nói chuyện
  - Phát hiện nhắn tin/gõ phím
  - Phát hiện điều chỉnh thiết bị trên xe

- **Cảnh báo thông minh**:
  - Cảnh báo theo mức độ nghiêm trọng
  - Ghi lại các tình huống nguy hiểm
  - Theo dõi lịch sử cảnh báo

- **API tích hợp**:
  - API RESTful để tích hợp với các hệ thống khác
  - Phân tích ảnh tải lên
  - Kiểm soát hệ thống từ xa

## Cài đặt

### Yêu cầu

- Python 3.8+
- Các thư viện bổ sung: xem `requirements.txt`

### Các bước cài đặt

1. Clone repository:

```bash
git clone https://github.com/phucmouse135/giamsatATGT.git
cd driver-monitoring
```

2. Cài đặt các thư viện phụ thuộc:

```bash
pip install -r requirements.txt
```

## Huấn luyện mô hình

### 1. Chuẩn bị dữ liệu

Cấu trúc thư mục dữ liệu cần được tổ chức như sau:

```
datasets/
  ├── drowsiness/
  │    ├── alert/   # Video/hình ảnh tài xế tỉnh táo
  │    └── drowsy/  # Video/hình ảnh tài xế buồn ngủ
  └── distraction/
       ├── focused/       # Tài xế tập trung
       ├── talking/       # Tài xế nói chuyện
       ├── phone_use/     # Tài xế dùng điện thoại
       ├── texting/       # Tài xế nhắn tin
       ├── radio_adjust/  # Tài xế điều chỉnh radio/thiết bị
       └── looking_away/  # Tài xế nhìn ra ngoài
```

Dữ liệu có thể ở định dạng video (mp4, avi) hoặc các thư mục chứa chuỗi hình ảnh.

### 2. Huấn luyện

#### Huấn luyện tự động

Sử dụng script `train_all.py` để tự động huấn luyện cả hai mô hình:

```bash
python train_all.py --drowsiness_data datasets/drowsiness --distraction_data datasets/distraction
```

Các tùy chọn bổ sung:
- `--models_dir`: Thư mục lưu mô hình (mặc định: `models/saved`)
- `--epochs`: Số lượng epoch huấn luyện (mặc định: 50)
- `--batch_size`: Kích thước batch (mặc định: 32)
- `--only_drowsiness`: Chỉ huấn luyện mô hình phát hiện buồn ngủ
- `--only_distraction`: Chỉ huấn luyện mô hình phát hiện mất tập trung

#### Huấn luyện thủ công

Huấn luyện từng mô hình riêng biệt:

1. Mô hình phát hiện buồn ngủ:
```bash
python train_drowsiness.py --data_dir datasets/drowsiness --models_dir models/saved
```

2. Mô hình phát hiện mất tập trung:
```bash
python train_distraction.py --data_dir datasets/distraction --models_dir models/saved
```

### 3. Đánh giá mô hình

Sau khi huấn luyện, các biểu đồ đánh giá và báo cáo hiệu suất sẽ được lưu vào thư mục `models/saved`:

- `drowsiness_training_history.png`: Biểu đồ huấn luyện mô hình phát hiện buồn ngủ
- `distraction_training_history.png`: Biểu đồ huấn luyện mô hình phát hiện mất tập trung

## Chạy hệ thống

### 1. Sử dụng giao diện

Chạy ứng dụng với camera (webcam mặc định):

```bash
python -m app.main --camera 0
```

Các tùy chọn:
- `--camera ID`: ID của camera (mặc định: 0)
- `--no-ui`: Không hiển thị giao diện người dùng
- `--record`: Ghi lại video
- `--output DIR`: Thư mục đầu ra (mặc định: output)

### 2. Sử dụng API

Khởi động API server:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

API sẽ chạy tại địa chỉ http://localhost:8000.
Tài liệu API tự động: http://localhost:8000/docs

#### Các API endpoint chính:

- `POST /monitoring/start`: Bắt đầu giám sát
- `POST /monitoring/stop`: Dừng giám sát
- `GET /monitoring/status`: Lấy trạng thái hệ thống
- `GET /alerts/recent`: Lấy danh sách cảnh báo gần đây
- `POST /camera/analyze`: Phân tích hình ảnh tải lên

### 3. Tích hợp vào code

```python
from models.deep_learning import DeepLearningModelManager

# Khởi tạo model manager
model_manager = DeepLearningModelManager(
    models_dir='models/saved',
    load_models=True
)

# Xử lý frame (giả sử đã có các thông số từ các bộ dò)
results = model_manager.process_frame(
    frame, face_landmarks, left_ear, right_ear, mar, roll, pitch, yaw
)

# Kiểm tra kết quả
if results['drowsiness']['is_drowsy']:
    print(f"Tài xế buồn ngủ! Điểm: {results['drowsiness']['score']}")

if results['distraction']['is_distracted']:
    print(f"Tài xế mất tập trung! Loại: {results['distraction']['class_name']}")
```

## Cấu trúc dự án

```
driver_monitoring/
├── app/                  # Ứng dụng chính
│   ├── __init__.py
│   ├── main.py           # Mã chính của ứng dụng
│   └── api.py            # API endpoints
├── models/               # Các mô hình phát hiện
│   ├── __init__.py
│   ├── face_detector.py
│   ├── drowsiness_detector.py
│   ├── distraction_detector.py
│   ├── dangerous_behavior_detector.py
│   └── deep_learning/    # Mô hình deep learning
│       ├── __init__.py
│       ├── data_preprocessing.py
│       ├── drowsiness_model.py
│       ├── distraction_model.py
│       └── model_manager.py
├── utils/                # Các tiện ích
│   ├── __init__.py
│   ├── alert_manager.py
│   ├── file_utils.py
│   └── visualization.py
├── train_drowsiness.py   # Script huấn luyện mô hình buồn ngủ
├── train_distraction.py  # Script huấn luyện mô hình mất tập trung
├── train_all.py          # Script huấn luyện tất cả mô hình
├── requirements.txt      # Các gói thư viện cần thiết
└── README.md             # Tài liệu
```

## Dữ liệu mẫu

Các bộ dữ liệu công khai có thể được sử dụng:

- **NTHU-DDD**: Bộ dữ liệu phát hiện buồn ngủ
  - Link: https://sites.google.com/view/drowsy-detection

- **DMD**: Bộ dữ liệu phát hiện mất tập trung
  - Link: https://github.com/rezagl/DMD

- **YawDD**: Bộ dữ liệu phát hiện ngáp
  - Link: https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset

## Liên hệ

Nếu có bất kỳ câu hỏi hoặc góp ý nào, vui lòng liên hệ qua email: phucchuot37@gmail.com