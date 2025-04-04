import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import glob
from tqdm import tqdm
import argparse
from tensorflow.keras.utils import to_categorical

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.deep_learning import (
    FaceDataPreprocessor,
    DistractionDetectorModel,
    DeepLearningModelManager
)

# Định nghĩa các lớp phân loại
DISTRACTION_CLASSES = [
    'tập_trung',
    'nói_chuyện',
    'điện_thoại',
    'nhắn_tin',
    'điều_chỉnh_radio',
    'nhìn_khác'
]

def load_dataset(dataset_dir, target_shape=(128, 128), sequence_length=20):
    """
    Tải dữ liệu từ thư mục dataset với cấu trúc:
    dataset_dir/
        focused/ - tài xế tập trung
        talking/ - tài xế nói chuyện
        phone_use/ - tài xế dùng điện thoại
        texting/ - tài xế nhắn tin
        radio_adjust/ - tài xế điều chỉnh radio
        looking_away/ - tài xế nhìn ra ngoài
    
    Args:
        dataset_dir: Đường dẫn đến thư mục dataset
        target_shape: Kích thước ảnh đầu ra
        sequence_length: Độ dài chuỗi cho LSTM
        
    Returns:
        data: Từ điển chứa dữ liệu đã xử lý
    """
    print(f"Đang tải dữ liệu từ {dataset_dir}...")
    
    # Ánh xạ tên thư mục sang chỉ số lớp
    class_mapping = {
        'focused': 0,  # tập_trung
        'talking': 1,  # nói_chuyện
        'phone_use': 2,  # điện_thoại
        'texting': 3,  # nhắn_tin
        'radio_adjust': 4,  # điều_chỉnh_radio
        'looking_away': 5  # nhìn_khác
    }
    
    # Khởi tạo bộ tiền xử lý
    preprocessor = FaceDataPreprocessor(
        face_img_size=target_shape,
        sequence_length=sequence_length
    )
    
    face_sequences = []
    eye_sequences = []
    numeric_features = []
    labels = []
    
    # Xử lý các thư mục
    for class_dir_name, class_idx in class_mapping.items():
        class_dir = os.path.join(dataset_dir, class_dir_name)
        
        if not os.path.exists(class_dir):
            print(f"Thư mục {class_dir} không tồn tại!")
            continue
        
        # Tìm tất cả tệp video hoặc hình ảnh
        video_files = glob.glob(os.path.join(class_dir, "*.mp4")) + \
                     glob.glob(os.path.join(class_dir, "*.avi"))
        image_folders = [d for d in glob.glob(os.path.join(class_dir, "*")) if os.path.isdir(d)]
        
        # Xử lý các tệp video
        for video_file in tqdm(video_files, desc=f"Xử lý video {class_dir_name}"):
            try:
                # Đọc video
                cap = cv2.VideoCapture(video_file)
                frames = []
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                
                cap.release()
                
                # Chỉ lấy một số frame đều đặn để tạo chuỗi
                if len(frames) > sequence_length:
                    frame_indices = np.linspace(0, len(frames) - 1, sequence_length, dtype=int)
                    frames = [frames[i] for i in frame_indices]
                
                # Xử lý từng frame
                processed_frames = []
                for frame in frames:
                    # Giả định có bộ dò khuôn mặt và các điểm đánh dấu
                    # Trong ứng dụng thực tế, bạn cần sử dụng FaceDetector thực sự
                    # Đây chỉ là code demo
                    
                    # Giả lập dữ liệu đầu vào
                    face_roi = cv2.resize(frame, target_shape)
                    left_eye_roi = cv2.resize(frame[50:80, 40:80], (64, 32))
                    right_eye_roi = cv2.resize(frame[50:80, 80:120], (64, 32))
                    mouth_roi = cv2.resize(frame[100:130, 60:100], (64, 32))
                    ear = 0.3
                    mar = 0.5
                    
                    # Mô phỏng góc nhìn khác nhau dựa trên lớp
                    if class_dir_name == 'looking_away':
                        yaw = np.random.uniform(-0.7, -0.4) if np.random.random() > 0.5 else np.random.uniform(0.4, 0.7)
                        pitch = np.random.uniform(-0.3, 0.3)
                    elif class_dir_name == 'phone_use' or class_dir_name == 'texting':
                        yaw = np.random.uniform(-0.2, 0.2)
                        pitch = np.random.uniform(0.3, 0.6)  # Nhìn xuống
                    else:
                        yaw = np.random.uniform(-0.2, 0.2)
                        pitch = np.random.uniform(-0.2, 0.2)
                    
                    roll = np.random.uniform(-0.1, 0.1)
                    head_pose = (roll, pitch, yaw)
                    
                    # Cập nhật bộ tiền xử lý
                    preprocessor.update_sequences(face_roi, left_eye_roi, right_eye_roi, mouth_roi, ear, mar, head_pose)
                
                # Lấy dữ liệu chuỗi từ bộ tiền xử lý
                sequence_data = preprocessor.get_sequence_data()
                
                # Thêm vào tập dữ liệu
                face_sequences.append(sequence_data['face_sequence'])
                eye_sequences.append(sequence_data['eye_sequence'])
                numeric_features.append(sequence_data['numeric_features'])
                labels.append(class_idx)
                
            except Exception as e:
                print(f"Lỗi khi xử lý {video_file}: {str(e)}")
        
        # Xử lý các thư mục hình ảnh tương tự...
    
    # Chuyển đổi thành mảng numpy
    data = {
        'face_sequence': np.array(face_sequences),
        'eye_sequence': np.array(eye_sequences),
        'numeric_features': np.array(numeric_features),
        'labels': to_categorical(np.array(labels), len(DISTRACTION_CLASSES))  # One-hot encoding
    }
    
    return data

def train_model(data_dir, models_dir='models/saved', epochs=50, batch_size=32):
    """
    Huấn luyện mô hình phát hiện mất tập trung
    
    Args:
        data_dir: Đường dẫn đến thư mục dữ liệu
        models_dir: Đường dẫn đến thư mục lưu mô hình
        epochs: Số lượng epoch huấn luyện
        batch_size: Kích thước batch
    """
    # Tạo thư mục lưu mô hình nếu chưa tồn tại
    os.makedirs(models_dir, exist_ok=True)
    
    # Tải dữ liệu
    data = load_dataset(data_dir)
    
    # Chia tập train/test
    train_ratio = 0.8
    indices = np.arange(len(data['labels']))
    
    # Tạo y để stratify (đảm bảo tỷ lệ các lớp giống nhau ở tập train và test)
    y = np.argmax(data['labels'], axis=1)
    
    train_indices, test_indices = train_test_split(indices, train_size=train_ratio, 
                                                 stratify=y, random_state=42)
    
    # Tạo tập huấn luyện và kiểm định
    train_data = {
        'face_sequence': data['face_sequence'][train_indices],
        'eye_sequence': data['eye_sequence'][train_indices],
        'numeric_features': data['numeric_features'][train_indices],
        'labels': data['labels'][train_indices]
    }
    
    test_data = {
        'face_sequence': data['face_sequence'][test_indices],
        'eye_sequence': data['eye_sequence'][test_indices],
        'numeric_features': data['numeric_features'][test_indices],
        'labels': data['labels'][test_indices]
    }
    
    # Khởi tạo model manager
    model_manager = DeepLearningModelManager(models_dir=models_dir)
    
    # Callback để lưu mô hình tốt nhất
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(models_dir, 'distraction_model_best.h5'),
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    
    # Callback để dừng sớm
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Huấn luyện mô hình
    print("Bắt đầu huấn luyện mô hình phát hiện mất tập trung...")
    history = model_manager.train_distraction_model(
        train_data=train_data,
        validation_data=test_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_callback, early_stopping],
        save_after_training=True
    )
    
    # Đánh giá mô hình
    print("Đánh giá mô hình phát hiện mất tập trung...")
    metrics = model_manager.evaluate_models(distraction_test_data=test_data)
    print(f"Kết quả: {metrics['distraction']}")
    
    # Vẽ đồ thị lịch sử huấn luyện
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Mô hình phát hiện mất tập trung - Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Mô hình phát hiện mất tập trung - Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'distraction_training_history.png'))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình phát hiện mất tập trung")
    parser.add_argument("--data_dir", type=str, required=True, help="Đường dẫn đến thư mục dữ liệu")
    parser.add_argument("--models_dir", type=str, default="models/saved", help="Đường dẫn đến thư mục lưu mô hình")
    parser.add_argument("--epochs", type=int, default=50, help="Số lượng epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch")
    
    args = parser.parse_args()
    
    train_model(args.data_dir, args.models_dir, args.epochs, args.batch_size) 