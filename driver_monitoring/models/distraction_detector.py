import numpy as np
import cv2
import math
from collections import deque
import time
import logging
import os
from typing import Dict, Any, Tuple, Optional, Union, List

class DistractionDetector:
    """
    Phát hiện trạng thái mất tập trung của tài xế
    """
    def __init__(self, yaw_threshold=0.4, pitch_threshold=0.3, duration_threshold=2.0, phone_duration=1.5, 
                 use_deep_learning=False, models_dir="models/saved"):
        # Ngưỡng góc quay đầu sang trái/phải để xác định mất tập trung
        self.yaw_threshold = yaw_threshold
        
        # Ngưỡng góc ngẩng/cúi đầu để xác định mất tập trung
        self.pitch_threshold = pitch_threshold
        
        # Thời gian tối thiểu (giây) mất tập trung để ghi nhận
        self.duration_threshold = duration_threshold
        
        # Thời gian tối thiểu (giây) nhìn điện thoại để ghi nhận
        self.phone_duration = phone_duration
        
        # Theo dõi thời gian mất tập trung hiện tại
        self.current_distraction_start = None
        
        # Theo dõi loại mất tập trung hiện tại
        self.current_distraction_type = None
        
        # Lịch sử các điểm nhìn gần đây (yaw, pitch) để phát hiện mẫu hình
        self.gaze_history = deque(maxlen=30)  # 1 giây @ 30fps
        
        # Điểm nhìn tham chiếu (hướng tới trước) cho tài xế
        self.reference_gaze = None
        
        # Số khung hình cần thiết để thiết lập điểm nhìn tham chiếu
        self.calibration_frames = 30
        
        # Cờ đánh dấu xem điểm nhìn tham chiếu đã được thiết lập chưa
        self.calibrated = False
        
        # Bộ đếm khung hình để hiệu chuẩn
        self.calibration_counter = 0
        
        # Dữ liệu hiệu chuẩn tạm thời
        self.temp_calibration_data = []
        
        # Sử dụng mô hình deep learning nếu được yêu cầu
        self.use_deep_learning = use_deep_learning
        self.dl_model_manager = None
        self.models_dir = models_dir
        
        # Ánh xạ các lớp từ mô hình deep learning sang định dạng hiện tại
        self.dl_class_to_distraction_type = {
            'tập_trung': None,
            'nói_chuyện': "talking",
            'điện_thoại': "using_phone",
            'nhắn_tin': "texting",
            'điều_chỉnh_radio': "adjusting_controls",
            'nhìn_khác': "looking_away"
        }
        
        # Cài đặt logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('DistractionDetector')
        
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
    
    def calibrate(self, yaw, pitch, roll):
        """
        Thêm dữ liệu vào quá trình hiệu chuẩn
        
        Args:
            yaw, pitch, roll: Các góc hướng đầu hiện tại
        
        Returns:
            calibrated: Đã hoàn thành hiệu chuẩn hay chưa
        """
        if not self.calibrated:
            self.temp_calibration_data.append((yaw, pitch, roll))
            self.calibration_counter += 1
            
            if self.calibration_counter >= self.calibration_frames:
                # Tính toán điểm nhìn tham chiếu bằng cách lấy trung bình
                yaws, pitches, rolls = zip(*self.temp_calibration_data)
                self.reference_gaze = (np.mean(yaws), np.mean(pitches), np.mean(rolls))
                self.calibrated = True
        
        return self.calibrated
    
    def reset_calibration(self):
        """Đặt lại quá trình hiệu chuẩn"""
        self.calibrated = False
        self.calibration_counter = 0
        self.temp_calibration_data = []
        self.reference_gaze = None
    
    def detect_phone_usage(self, frame, face_landmarks):
        """
        Phát hiện sử dụng điện thoại thông qua phân tích tư thế tay và đầu
        
        Args:
            frame: Khung hình hiện tại
            face_landmarks: Các điểm đánh dấu khuôn mặt
        
        Returns:
            using_phone: Boolean chỉ ra tài xế có đang sử dụng điện thoại không
        """
        # Đây là một phương pháp đơn giản để demo
        # Trong triển khai thực tế, chúng ta sẽ sử dụng mô hình phát hiện đối tượng
        # để phát hiện điện thoại và vị trí tay
        
        # Giả sử chúng ta có thể xác định vị trí tay và điện thoại
        # Phương pháp đơn giản: Kiểm tra điểm nhìn xuống (cúi đầu)
        
        # Trong demo này, chúng ta chỉ sử dụng góc pitch của đầu
        # pitch > 0 khi nhìn xuống
        if self.reference_gaze and self.current_distraction_type == "looking_down":
            return True
        
        return False
    
    def detect_distraction(self, yaw, pitch, roll, frame=None, face_landmarks=None, left_ear=None, right_ear=None, mar=None):
        """
        Phát hiện mất tập trung dựa trên hướng đầu
        
        Args:
            yaw, pitch, roll: Các góc hướng đầu hiện tại
            frame: Khung hình hiện tại (tùy chọn, cho phát hiện điện thoại)
            face_landmarks: Các điểm đánh dấu khuôn mặt (tùy chọn)
            left_ear, right_ear: Tỷ lệ khía cạnh mắt trái và phải (tùy chọn, cho deep learning)
            mar: Tỷ lệ khía cạnh miệng (tùy chọn, cho deep learning)
            
        Returns:
            distracted: Boolean chỉ ra tài xế có mất tập trung không
            distraction_type: Loại mất tập trung
            distraction_level: Mức độ mất tập trung (0-1)
        """
        # Nếu sử dụng deep learning và có đủ dữ liệu đầu vào
        if self.use_deep_learning and self.dl_model_manager and frame is not None and face_landmarks is not None:
            try:
                # Xử lý frame bằng mô hình deep learning
                dl_results = self.dl_model_manager.process_frame(
                    frame, face_landmarks, left_ear or 0.3, right_ear or 0.3, mar or 0.5, roll, pitch, yaw
                )
                
                # Lấy kết quả phát hiện mất tập trung từ mô hình deep learning
                distracted = dl_results['distraction']['is_distracted']
                dl_class_name = dl_results['distraction']['class_name']
                confidence = dl_results['distraction']['confidence']
                
                # Chuyển đổi tên lớp từ mô hình deep learning sang định dạng hiện tại
                distraction_type = self.dl_class_to_distraction_type.get(dl_class_name)
                
                if distracted and distraction_type:
                    # Cập nhật thời gian bắt đầu mất tập trung nếu cần
                    current_time = time.time()
                    if self.current_distraction_start is None or self.current_distraction_type != distraction_type:
                        self.current_distraction_start = current_time
                        self.current_distraction_type = distraction_type
                    
                    # Tính thời gian mất tập trung
                    distraction_duration = current_time - self.current_distraction_start
                    
                    # Điều chỉnh mức độ mất tập trung dựa trên thời gian và điểm tin cậy
                    base_level = confidence
                    duration_factor = min(1.0, distraction_duration / (self.duration_threshold * 3))
                    distraction_level = base_level * (0.7 + 0.3 * duration_factor)
                    
                    return distracted, distraction_type, distraction_level
                else:
                    # Đặt lại nếu không phát hiện mất tập trung
                    self.current_distraction_start = None
                    self.current_distraction_type = None
                    return False, None, 0.0
                
            except Exception as e:
                self.logger.error(f"Lỗi khi sử dụng mô hình deep learning, chuyển sang phương pháp truyền thống: {str(e)}")
                # Nếu có lỗi, sử dụng phương pháp truyền thống
        
        # Nếu chưa hiệu chuẩn, không thể phát hiện mất tập trung
        if not self.calibrated:
            return False, None, 0.0
        
        # Theo dõi thời gian
        current_time = time.time()
        
        # Thêm các hướng nhìn hiện tại vào lịch sử
        self.gaze_history.append((yaw, pitch, roll))
        
        # Tính độ lệch từ hướng nhìn tham chiếu
        ref_yaw, ref_pitch, ref_roll = self.reference_gaze
        yaw_deviation = abs(yaw - ref_yaw)
        pitch_deviation = abs(pitch - ref_pitch)
        
        # Xác định loại mất tập trung
        distraction_type = None
        
        if yaw_deviation > self.yaw_threshold:
            if yaw > ref_yaw:
                distraction_type = "looking_right"
            else:
                distraction_type = "looking_left"
        elif pitch_deviation > self.pitch_threshold:
            if pitch > ref_pitch:
                distraction_type = "looking_down"
            else:
                distraction_type = "looking_up"
        
        # Kiểm tra sử dụng điện thoại
        using_phone = False
        if frame is not None and face_landmarks is not None and distraction_type == "looking_down":
            using_phone = self.detect_phone_usage(frame, face_landmarks)
            if using_phone:
                distraction_type = "using_phone"
        
        # Nếu phát hiện mất tập trung
        if distraction_type:
            # Nếu đây là sự bắt đầu của một trạng thái mất tập trung mới
            if self.current_distraction_start is None or self.current_distraction_type != distraction_type:
                self.current_distraction_start = current_time
                self.current_distraction_type = distraction_type
            
            # Tính thời gian mất tập trung
            distraction_duration = current_time - self.current_distraction_start
            
            # Xác định ngưỡng thời gian dựa trên loại mất tập trung
            threshold = self.phone_duration if distraction_type == "using_phone" else self.duration_threshold
            
            # Nếu đủ thời gian, báo cáo mất tập trung
            if distraction_duration >= threshold:
                # Tính mức độ mất tập trung dựa trên thời lượng và độ lệch
                max_duration = threshold * 3  # Mức độ tối đa sau 3 lần ngưỡng
                duration_factor = min(1.0, distraction_duration / max_duration)
                
                # Kết hợp các yếu tố để có mức độ mất tập trung từ 0-1
                deviation_factor = max(yaw_deviation / (self.yaw_threshold * 2), 
                                     pitch_deviation / (self.pitch_threshold * 2))
                
                distraction_level = 0.3 + 0.7 * ((duration_factor + deviation_factor) / 2)
                distraction_level = min(1.0, distraction_level)
                
                return True, distraction_type, distraction_level
        else:
            # Đặt lại nếu không phát hiện mất tập trung
            self.current_distraction_start = None
            self.current_distraction_type = None
        
        return False, None, 0.0
        
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
    
    def get_distraction_types(self) -> Dict[str, str]:
        """
        Lấy danh sách các loại mất tập trung được hỗ trợ
        
        Returns:
            distraction_types: Từ điển các loại mất tập trung và mô tả
        """
        if self.use_deep_learning and self.dl_model_manager:
            dl_classes = self.dl_model_manager.get_distraction_classes()
            # Ánh xạ các lớp từ mô hình deep learning
            return {
                'tập_trung': "Tài xế đang tập trung lái xe",
                'nói_chuyện': "Tài xế đang nói chuyện với hành khách",
                'điện_thoại': "Tài xế đang sử dụng điện thoại",
                'nhắn_tin': "Tài xế đang nhắn tin/nhập liệu",
                'điều_chỉnh_radio': "Tài xế đang điều chỉnh thiết bị trên xe",
                'nhìn_khác': "Tài xế đang nhìn ra ngoài/không tập trung"
            }
        else:
            # Các loại mất tập trung truyền thống
            return {
                "looking_left": "Tài xế đang nhìn sang trái",
                "looking_right": "Tài xế đang nhìn sang phải",
                "looking_up": "Tài xế đang nhìn lên trên",
                "looking_down": "Tài xế đang nhìn xuống dưới",
                "using_phone": "Tài xế đang sử dụng điện thoại"
            } 