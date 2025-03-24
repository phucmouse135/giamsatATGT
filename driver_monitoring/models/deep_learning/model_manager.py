import os
import numpy as np
import cv2
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import threading
import logging
from datetime import datetime

from .drowsiness_model import DrowsinessDetectorModel
from .distraction_model import DistractionDetectorModel
from ..data_preprocessing import FaceDataPreprocessor

class DeepLearningModelManager:
    """
    Lớp quản lý các mô hình deep learning cho giám sát tài xế
    """
    def __init__(self, 
                 models_dir: str = 'models/saved',
                 use_pretrained: bool = True,
                 load_models: bool = False,
                 face_shape: Tuple[int, int] = (128, 128),
                 eye_shape: Tuple[int, int] = (32, 128),
                 mouth_shape: Tuple[int, int] = (32, 64),
                 sequence_length: int = 20):
        """
        Khởi tạo manager
        
        Args:
            models_dir: Thư mục chứa các mô hình đã lưu
            use_pretrained: Có sử dụng mô hình pretrained cho CNN hay không
            load_models: Có tải các mô hình đã lưu hay không
            face_shape: Kích thước ảnh khuôn mặt
            eye_shape: Kích thước ảnh mắt
            mouth_shape: Kích thước ảnh miệng
            sequence_length: Độ dài chuỗi cho LSTM
        """
        self.models_dir = models_dir
        self.use_pretrained = use_pretrained
        self.face_shape = face_shape
        self.eye_shape = eye_shape
        self.mouth_shape = mouth_shape
        self.sequence_length = sequence_length
        
        # Khởi tạo bộ tiền xử lý dữ liệu
        self.preprocessor = FaceDataPreprocessor(
            face_img_size=face_shape,
            sequence_length=sequence_length,
            use_landmarks=True,
            normalize=True
        )
        
        # Khởi tạo các mô hình
        self.drowsiness_model = DrowsinessDetectorModel(
            face_shape=face_shape,
            eye_shape=eye_shape,
            mouth_shape=mouth_shape,
            sequence_length=sequence_length,
            use_pretrained=use_pretrained
        )
        
        self.distraction_model = DistractionDetectorModel(
            face_shape=face_shape,
            eye_shape=eye_shape,
            sequence_length=sequence_length,
            use_pretrained=use_pretrained
        )
        
        # Đảm bảo thư mục mô hình tồn tại
        os.makedirs(models_dir, exist_ok=True)
        
        # Đường dẫn đến các mô hình đã lưu
        self.drowsiness_model_path = os.path.join(models_dir, 'drowsiness_model.h5')
        self.distraction_model_path = os.path.join(models_dir, 'distraction_model.h5')
        
        # Tải mô hình nếu cần
        if load_models:
            self.load_models()
        
        # Khởi tạo các biến theo dõi
        self.is_active = False
        self.detection_thread = None
        self.last_drowsiness_score = 0.0
        self.last_distraction_class = None
        self.last_distraction_confidence = 0.0
        self.last_update_time = None
        
        # Cài đặt logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('driver_monitoring.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DeepLearningModelManager')
    
    def load_models(self) -> bool:
        """
        Tải các mô hình đã lưu
        
        Returns:
            success: True nếu tải thành công, False nếu không
        """
        try:
            if os.path.exists(self.drowsiness_model_path):
                self.drowsiness_model.load(self.drowsiness_model_path)
                self.logger.info(f"Đã tải mô hình buồn ngủ từ {self.drowsiness_model_path}")
            else:
                self.logger.warning(f"Không tìm thấy mô hình buồn ngủ tại {self.drowsiness_model_path}")
                return False
            
            if os.path.exists(self.distraction_model_path):
                self.distraction_model.load(self.distraction_model_path)
                self.logger.info(f"Đã tải mô hình mất tập trung từ {self.distraction_model_path}")
            else:
                self.logger.warning(f"Không tìm thấy mô hình mất tập trung tại {self.distraction_model_path}")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi tải mô hình: {str(e)}")
            return False
    
    def save_models(self) -> bool:
        """
        Lưu các mô hình
        
        Returns:
            success: True nếu lưu thành công, False nếu không
        """
        try:
            self.drowsiness_model.save(self.drowsiness_model_path)
            self.logger.info(f"Đã lưu mô hình buồn ngủ tại {self.drowsiness_model_path}")
            
            self.distraction_model.save(self.distraction_model_path)
            self.logger.info(f"Đã lưu mô hình mất tập trung tại {self.distraction_model_path}")
            
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu mô hình: {str(e)}")
            return False
    
    def process_frame(self, 
                      frame: np.ndarray, 
                      face_landmarks, 
                      left_ear: float, 
                      right_ear: float,
                      mar: float,
                      roll: float,
                      pitch: float,
                      yaw: float) -> Dict[str, Any]:
        """
        Xử lý một khung hình và trả về kết quả dự đoán
        
        Args:
            frame: Khung hình đầu vào
            face_landmarks: Các điểm đánh dấu khuôn mặt
            left_ear, right_ear: Tỷ lệ khía cạnh mắt trái và phải
            mar: Tỷ lệ khía cạnh miệng
            roll, pitch, yaw: Góc nghiêng, ngẩng và quay đầu
            
        Returns:
            results: Từ điển chứa kết quả dự đoán
        """
        # Tiền xử lý dữ liệu
        processed_data = self.preprocessor.process_frame(
            frame, face_landmarks, left_ear, right_ear, mar, roll, pitch, yaw
        )
        
        # Dữ liệu đầu vào cho mô hình buồn ngủ
        drowsiness_input = {
            'face_sequence': np.expand_dims(processed_data['sequences']['face_sequence'], axis=0),
            'eye_sequence': np.expand_dims(processed_data['sequences']['eye_sequence'], axis=0),
            'mouth_sequence': np.expand_dims(processed_data['sequences']['mouth_sequence'], axis=0),
            'numeric_features': np.expand_dims(processed_data['sequences']['numeric_features'], axis=0)
        }
        
        # Dữ liệu đầu vào cho mô hình mất tập trung
        head_pose_sequence = np.array([pose for pose in processed_data['sequences']['numeric_features'][:, 2:5]])
        
        distraction_input = {
            'face_sequence': np.expand_dims(processed_data['sequences']['face_sequence'], axis=0),
            'eye_sequence': np.expand_dims(processed_data['sequences']['eye_sequence'], axis=0),
            'head_pose': np.expand_dims(head_pose_sequence, axis=0)
        }
        
        # Dự đoán buồn ngủ
        try:
            drowsiness_score = self.drowsiness_model.predict(drowsiness_input)[0][0]
            self.last_drowsiness_score = float(drowsiness_score)
        except Exception as e:
            self.logger.error(f"Lỗi khi dự đoán buồn ngủ: {str(e)}")
            drowsiness_score = self.last_drowsiness_score
        
        # Dự đoán mất tập trung
        try:
            distraction_class, class_idx, confidence = self.distraction_model.predict_class(distraction_input)
            self.last_distraction_class = distraction_class
            self.last_distraction_confidence = float(confidence)
        except Exception as e:
            self.logger.error(f"Lỗi khi dự đoán mất tập trung: {str(e)}")
            distraction_class = self.last_distraction_class
            confidence = self.last_distraction_confidence
            class_idx = -1
        
        # Cập nhật thời gian
        self.last_update_time = datetime.now()
        
        # Kết quả
        results = {
            'drowsiness': {
                'score': float(drowsiness_score),
                'is_drowsy': float(drowsiness_score) > 0.7,  # Ngưỡng buồn ngủ
            },
            'distraction': {
                'class_name': distraction_class,
                'class_idx': int(class_idx) if class_idx != -1 else None,
                'confidence': float(confidence),
                'is_distracted': distraction_class != 'tập_trung' and float(confidence) > 0.6  # Ngưỡng mất tập trung
            },
            'timestamp': self.last_update_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        }
        
        return results
    
    def start_detection(self, callback=None) -> bool:
        """
        Bắt đầu luồng phát hiện
        
        Args:
            callback: Hàm callback được gọi mỗi khi có kết quả dự đoán mới
            
        Returns:
            success: True nếu bắt đầu thành công, False nếu không
        """
        if self.is_active:
            self.logger.warning("Luồng phát hiện đã chạy.")
            return False
        
        self.is_active = True
        self.detection_thread = threading.Thread(target=self._detection_loop, args=(callback,))
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        self.logger.info("Đã bắt đầu luồng phát hiện.")
        return True
    
    def stop_detection(self) -> bool:
        """
        Dừng luồng phát hiện
        
        Returns:
            success: True nếu dừng thành công, False nếu không
        """
        if not self.is_active:
            self.logger.warning("Luồng phát hiện không chạy.")
            return False
        
        self.is_active = False
        if self.detection_thread:
            self.detection_thread.join(timeout=5.0)
            if self.detection_thread.is_alive():
                self.logger.warning("Không thể kết thúc luồng phát hiện đúng cách.")
                return False
        
        self.logger.info("Đã dừng luồng phát hiện.")
        return True
    
    def _detection_loop(self, callback=None) -> None:
        """
        Vòng lặp phát hiện chạy trong một luồng riêng
        
        Args:
            callback: Hàm callback được gọi mỗi khi có kết quả dự đoán mới
        """
        # Vòng lặp này sẽ được triển khai khi tích hợp với hệ thống camera thực tế
        # Hiện tại, nó chỉ là một phương thức giả để hoàn thiện API
        self.logger.info("Đã bắt đầu vòng lặp phát hiện.")
        
        while self.is_active:
            time.sleep(1)  # Chờ cho phiên bản thực tế
            
        self.logger.info("Đã dừng vòng lặp phát hiện.")
    
    def train_drowsiness_model(self, 
                               train_data: Dict[str, np.ndarray],
                               validation_data: Dict[str, np.ndarray] = None,
                               epochs: int = 50,
                               batch_size: int = 32,
                               save_after_training: bool = True) -> tf.keras.callbacks.History:
        """
        Huấn luyện mô hình phát hiện buồn ngủ
        
        Args:
            train_data: Dữ liệu huấn luyện
            validation_data: Dữ liệu kiểm định
            epochs: Số lượng epoch
            batch_size: Kích thước batch
            save_after_training: Có lưu mô hình sau khi huấn luyện hay không
            
        Returns:
            history: Lịch sử huấn luyện
        """
        self.logger.info(f"Bắt đầu huấn luyện mô hình buồn ngủ với {epochs} epochs")
        
        # Tạo callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'logs/drowsiness_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                histogram_freq=1
            )
        ]
        
        # Huấn luyện mô hình
        history = self.drowsiness_model.train(
            train_data=train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        # Lưu mô hình nếu cần
        if save_after_training:
            self.drowsiness_model.save(self.drowsiness_model_path)
            self.logger.info(f"Đã lưu mô hình buồn ngủ tại {self.drowsiness_model_path}")
        
        self.logger.info("Đã hoàn thành huấn luyện mô hình buồn ngủ")
        return history
    
    def train_distraction_model(self, 
                                train_data: Dict[str, np.ndarray],
                                validation_data: Dict[str, np.ndarray] = None,
                                epochs: int = 50,
                                batch_size: int = 32,
                                save_after_training: bool = True) -> tf.keras.callbacks.History:
        """
        Huấn luyện mô hình phát hiện mất tập trung
        
        Args:
            train_data: Dữ liệu huấn luyện
            validation_data: Dữ liệu kiểm định
            epochs: Số lượng epoch
            batch_size: Kích thước batch
            save_after_training: Có lưu mô hình sau khi huấn luyện hay không
            
        Returns:
            history: Lịch sử huấn luyện
        """
        self.logger.info(f"Bắt đầu huấn luyện mô hình mất tập trung với {epochs} epochs")
        
        # Tạo callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'logs/distraction_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                histogram_freq=1
            )
        ]
        
        # Huấn luyện mô hình
        history = self.distraction_model.train(
            train_data=train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        # Lưu mô hình nếu cần
        if save_after_training:
            self.distraction_model.save(self.distraction_model_path)
            self.logger.info(f"Đã lưu mô hình mất tập trung tại {self.distraction_model_path}")
        
        self.logger.info("Đã hoàn thành huấn luyện mô hình mất tập trung")
        return history
    
    def evaluate_models(self, 
                        drowsiness_test_data: Dict[str, np.ndarray] = None,
                        distraction_test_data: Dict[str, np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """
        Đánh giá cả hai mô hình
        
        Args:
            drowsiness_test_data: Dữ liệu kiểm tra cho mô hình buồn ngủ
            distraction_test_data: Dữ liệu kiểm tra cho mô hình mất tập trung
            
        Returns:
            results: Từ điển chứa các metric đánh giá cho cả hai mô hình
        """
        results = {}
        
        if drowsiness_test_data is not None:
            self.logger.info("Đánh giá mô hình buồn ngủ")
            drowsiness_metrics = self.drowsiness_model.evaluate(drowsiness_test_data)
            results['drowsiness'] = drowsiness_metrics
            
            self.logger.info(f"Kết quả đánh giá mô hình buồn ngủ: {drowsiness_metrics}")
        
        if distraction_test_data is not None:
            self.logger.info("Đánh giá mô hình mất tập trung")
            distraction_metrics = self.distraction_model.evaluate(distraction_test_data)
            results['distraction'] = distraction_metrics
            
            self.logger.info(f"Kết quả đánh giá mô hình mất tập trung: {distraction_metrics}")
        
        return results
    
    def get_model_summaries(self) -> Dict[str, List[str]]:
        """
        Lấy tóm tắt của các mô hình
        
        Returns:
            summaries: Từ điển chứa tóm tắt mô hình dưới dạng danh sách các dòng
        """
        # Sử dụng StringIO để capture output từ summary
        import io
        from contextlib import redirect_stdout
        
        drowsiness_summary = io.StringIO()
        with redirect_stdout(drowsiness_summary):
            self.drowsiness_model.summary()
        
        distraction_summary = io.StringIO()
        with redirect_stdout(distraction_summary):
            self.distraction_model.summary()
        
        return {
            'drowsiness': drowsiness_summary.getvalue().strip().split('\n'),
            'distraction': distraction_summary.getvalue().strip().split('\n')
        }
    
    def get_distraction_classes(self) -> List[str]:
        """
        Lấy danh sách các lớp mất tập trung
        
        Returns:
            classes: Danh sách các lớp mất tập trung
        """
        return self.distraction_model.get_class_names() 