import numpy as np
import time
from collections import deque
import os
import logging
from typing import Dict, Any, Tuple, Optional, Union

class DrowsinessDetector:
    """
    Phát hiện trạng thái buồn ngủ của tài xế dựa trên các đặc trưng khuôn mặt
    """
    def __init__(self, ear_threshold=0.25, ear_consec_frames=20, mar_threshold=0.6, head_movement_threshold=0.3, 
                 use_deep_learning=False, models_dir="models/saved"):
        # Ngưỡng tỷ lệ khía cạnh mắt (EAR) để phát hiện mắt nhắm
        self.ear_threshold = ear_threshold
        
        # Số khung hình liên tiếp mắt nhắm để xác định buồn ngủ
        self.ear_consec_frames = ear_consec_frames
        
        # Ngưỡng tỷ lệ khía cạnh miệng (MAR) để phát hiện ngáp
        self.mar_threshold = mar_threshold
        
        # Ngưỡng chuyển động đầu để phát hiện gật gù
        self.head_movement_threshold = head_movement_threshold
        
        # Bộ đếm số khung hình mắt nhắm liên tiếp
        self.eye_closed_counter = 0
        
        # Bộ đếm số lần ngáp
        self.yawn_counter = 0
        
        # Thời gian bắt đầu ngáp
        self.yawn_start_time = None
        
        # Theo dõi các góc nghiêng đầu gần đây
        self.pitch_history = deque(maxlen=30)  # Khoảng 1 giây ở 30 FPS
        
        # Trạng thái ngáp
        self.is_yawning = False
        
        # Thời gian bắt đầu chớp mắt
        self.blink_start_time = None
        
        # Đếm số lần chớp mắt trong khoảng thời gian
        self.blink_counter = 0
        self.blink_time_window = 60  # Số giây để đếm số lần chớp mắt
        self.blink_start_window = time.time()
        
        # Ngưỡng số lần chớp mắt bất thường
        self.normal_blink_rate = (15, 30)  # (min, max) số lần chớp mắt bình thường trong 1 phút
        
        # Sử dụng mô hình deep learning nếu được yêu cầu
        self.use_deep_learning = use_deep_learning
        self.dl_model_manager = None
        self.models_dir = models_dir
        
        # Cài đặt logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('DrowsinessDetector')
        
        # Khởi tạo model manager nếu sử dụng deep learning
        if self.use_deep_learning:
            self._init_deep_learning_model()
            
    def _init_deep_learning_model(self):
        """
        Khởi tạo mô hình deep learning nếu được yêu cầu
        """
        try:
            # Import động để tránh lỗi nếu không có các thư viện deep learning
            from ..models.deep_learning.model_manager import DeepLearningModelManager
            
            # Khởi tạo model manager
            self.dl_model_manager = DeepLearningModelManager(
                models_dir=self.models_dir,
                use_pretrained=True,
                load_models=True  # Cố gắng tải mô hình đã lưu
            )
            
            self.logger.info("Đã khởi tạo mô hình deep learning")
        except Exception as e:
            self.logger.error(f"Không thể khởi tạo mô hình deep learning: {str(e)}")
            self.use_deep_learning = False
    
    def detect_drowsiness(self, left_ear, right_ear, mar, pitch, yaw, roll, 
                           frame=None, face_landmarks=None) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Phát hiện buồn ngủ từ các thông số khuôn mặt
        
        Args:
            left_ear: Tỷ lệ khía cạnh mắt trái
            right_ear: Tỷ lệ khía cạnh mắt phải
            mar: Tỷ lệ khía cạnh miệng
            pitch, yaw, roll: Góc nghiêng, quay và lăn của đầu
            frame: Khung hình đầu vào (tùy chọn, cần nếu sử dụng deep learning)
            face_landmarks: Các điểm đánh dấu khuôn mặt (tùy chọn, cần nếu sử dụng deep learning)
            
        Returns:
            drowsy: Boolean chỉ ra tài xế có buồn ngủ không
            drowsy_level: Mức độ buồn ngủ (0-1)
            drowsiness_indicators: Từ điển các chỉ số buồn ngủ
        """
        # Nếu sử dụng deep learning và có đủ dữ liệu đầu vào
        if self.use_deep_learning and self.dl_model_manager and frame is not None and face_landmarks is not None:
            try:
                # Xử lý frame bằng mô hình deep learning
                dl_results = self.dl_model_manager.process_frame(
                    frame, face_landmarks, left_ear, right_ear, mar, roll, pitch, yaw
                )
                
                # Lấy kết quả phát hiện buồn ngủ từ mô hình deep learning
                drowsy = dl_results['drowsiness']['is_drowsy']
                drowsy_level = dl_results['drowsiness']['score']
                
                # Bổ sung các chỉ số truyền thống cho kết quả
                drowsiness_indicators = self._traditional_drowsiness_indicators(left_ear, right_ear, mar, pitch, yaw, roll)
                
                # Bổ sung thông tin từ mô hình deep learning
                drowsiness_indicators['dl_score'] = drowsy_level
                
                return drowsy, drowsy_level, drowsiness_indicators
                
            except Exception as e:
                self.logger.error(f"Lỗi khi sử dụng mô hình deep learning, chuyển sang phương pháp truyền thống: {str(e)}")
                # Nếu có lỗi, sử dụng phương pháp truyền thống
        
        # Phương pháp truyền thống
        return self._traditional_drowsiness_detection(left_ear, right_ear, mar, pitch, yaw, roll)
    
    def _traditional_drowsiness_indicators(self, left_ear, right_ear, mar, pitch, yaw, roll):
        """
        Tính toán các chỉ số buồn ngủ sử dụng phương pháp truyền thống
        
        Args:
            left_ear, right_ear: Tỷ lệ khía cạnh mắt
            mar: Tỷ lệ khía cạnh miệng
            pitch, yaw, roll: Góc nghiêng, quay và lăn của đầu
            
        Returns:
            drowsiness_indicators: Từ điển các chỉ số buồn ngủ
        """
        # Tính trung bình EAR của cả hai mắt
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Cập nhật lịch sử góc nghiêng đầu
        self.pitch_history.append(pitch)
        
        # Theo dõi thời gian
        current_time = time.time()
        
        # Phát hiện buồn ngủ dựa trên các chỉ số
        drowsiness_indicators = {
            "eyes_closed": False,
            "yawning": False,
            "head_nodding": False,
            "abnormal_blink_rate": False
        }
        
        # Phát hiện mắt nhắm
        if avg_ear < self.ear_threshold:
            self.eye_closed_counter += 1
            
            # Bắt đầu đếm thời gian chớp mắt
            if self.blink_start_time is None:
                self.blink_start_time = current_time
        else:
            # Nếu mắt mở lại sau khi nhắm
            if self.eye_closed_counter >= 3:  # Chớp mắt thông thường kéo dài 100-400ms ở 30fps
                self.blink_counter += 1
                
            # Reset bộ đếm và thời gian bắt đầu
            self.eye_closed_counter = 0
            self.blink_start_time = None
        
        # Kiểm tra mắt nhắm kéo dài
        if self.eye_closed_counter >= self.ear_consec_frames:
            drowsiness_indicators["eyes_closed"] = True
        
        # Phát hiện ngáp dựa trên MAR
        if mar > self.mar_threshold:
            if not self.is_yawning:
                self.is_yawning = True
                self.yawn_start_time = current_time
        else:
            if self.is_yawning:
                self.is_yawning = False
                if self.yawn_start_time is not None:
                    yawn_duration = current_time - self.yawn_start_time
                    
                    # Ngáp kéo dài hơn 2 giây được coi là một dấu hiệu buồn ngủ
                    if yawn_duration > 2.0:
                        self.yawn_counter += 1
                        drowsiness_indicators["yawning"] = True
        
        # Phát hiện gật đầu dựa trên thay đổi góc nghiêng
        if len(self.pitch_history) >= 2:
            pitch_changes = np.diff(list(self.pitch_history))
            max_pitch_change = np.max(np.abs(pitch_changes))
            
            if max_pitch_change > self.head_movement_threshold:
                drowsiness_indicators["head_nodding"] = True
        
        # Kiểm tra tốc độ chớp mắt
        elapsed_time = current_time - self.blink_start_window
        if elapsed_time >= self.blink_time_window:
            blink_rate = (self.blink_counter / elapsed_time) * 60  # Chuyển đổi thành bpm
            
            # Kiểm tra tốc độ chớp mắt bất thường
            if blink_rate < self.normal_blink_rate[0] or blink_rate > self.normal_blink_rate[1]:
                drowsiness_indicators["abnormal_blink_rate"] = True
            
            # Reset bộ đếm và thời gian bắt đầu cho cửa sổ mới
            self.blink_counter = 0
            self.blink_start_window = current_time
            
        return drowsiness_indicators
    
    def _traditional_drowsiness_detection(self, left_ear, right_ear, mar, pitch, yaw, roll):
        """
        Phát hiện buồn ngủ sử dụng phương pháp truyền thống
        
        Args:
            left_ear, right_ear: Tỷ lệ khía cạnh mắt
            mar: Tỷ lệ khía cạnh miệng
            pitch, yaw, roll: Góc nghiêng, quay và lăn của đầu
            
        Returns:
            drowsy: Boolean chỉ ra tài xế có buồn ngủ không
            drowsy_level: Mức độ buồn ngủ (0-1)
            drowsiness_indicators: Từ điển các chỉ số buồn ngủ
        """
        # Lấy các chỉ số buồn ngủ
        drowsiness_indicators = self._traditional_drowsiness_indicators(left_ear, right_ear, mar, pitch, yaw, roll)
        
        # Xác định mức độ buồn ngủ dựa trên số lượng chỉ số
        drowsy_indicators_count = sum(drowsiness_indicators.values())
        drowsy_level = min(1.0, drowsy_indicators_count / 4.0)  # Tối đa 4 chỉ số
        
        # Tài xế được coi là buồn ngủ nếu ít nhất một chỉ số là dương
        drowsy = drowsy_level > 0
        
        return drowsy, drowsy_level, drowsiness_indicators
    
    def set_use_deep_learning(self, use_deep_learning: bool) -> bool:
        """
        Bật/tắt sử dụng mô hình deep learning
        
        Args:
            use_deep_learning: Có sử dụng mô hình deep learning hay không
            
        Returns:
            success: Có hoàn thành thay đổi hay không
        """
        if use_deep_learning == self.use_deep_learning:
            return True
            
        # Nếu bật deep learning
        if use_deep_learning:
            self.use_deep_learning = True
            self._init_deep_learning_model()
            return self.dl_model_manager is not None
        else:
            # Tắt deep learning
            self.use_deep_learning = False
            return True 