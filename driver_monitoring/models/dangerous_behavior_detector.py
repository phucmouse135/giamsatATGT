import numpy as np
import cv2
import time
from collections import deque

class DangerousBehaviorDetector:
    """
    Phát hiện các hành vi nguy hiểm của tài xế như hút thuốc, ăn uống khi lái xe
    """
    def __init__(self, detection_threshold=0.7, duration_threshold=1.5):
        # Ngưỡng xác suất để xác định hành vi
        self.detection_threshold = detection_threshold
        
        # Thời gian tối thiểu (giây) để xác nhận hành vi nguy hiểm
        self.duration_threshold = duration_threshold
        
        # Theo dõi thời gian hành vi nguy hiểm hiện tại
        self.current_behavior_start = None
        
        # Theo dõi loại hành vi nguy hiểm hiện tại
        self.current_behavior_type = None
        
        # Lịch sử các dự đoán gần đây
        self.prediction_history = {
            "smoking": deque(maxlen=10),
            "eating": deque(maxlen=10),
            "drinking": deque(maxlen=10),
            "calling": deque(maxlen=10)
        }
        
        # Định nghĩa các vùng vòng miệng và tay
        # Trong triển khai thực tế, sẽ sử dụng mô hình phát hiện đối tượng và tư thế
        self.mouth_region = [(0.45, 0.55), (0.55, 0.65)]  # (xmin, ymin), (xmax, ymax) tỷ lệ
        self.hand_regions = [
            [(0.0, 0.4), (0.4, 0.8)],     # Vùng tay trái
            [(0.6, 0.4), (1.0, 0.8)]      # Vùng tay phải
        ]
    
    def _get_motion_in_region(self, prev_frame, curr_frame, region, frame_shape):
        """
        Phát hiện chuyển động trong một vùng cụ thể
        
        Args:
            prev_frame: Khung hình trước đó
            curr_frame: Khung hình hiện tại
            region: Vùng quan tâm [(xmin, ymin), (xmax, ymax)] dưới dạng tỷ lệ
            frame_shape: Kích thước khung hình để chuyển đổi tỷ lệ thành pixel
            
        Returns:
            motion_score: Điểm số chuyển động trong vùng (0-1)
        """
        # Chuyển đổi vùng từ tỷ lệ sang pixel
        h, w = frame_shape[:2]
        xmin = int(region[0][0] * w)
        ymin = int(region[0][1] * h)
        xmax = int(region[1][0] * w)
        ymax = int(region[1][1] * h)
        
        # Trích xuất vùng từ các khung hình
        prev_roi = prev_frame[ymin:ymax, xmin:xmax]
        curr_roi = curr_frame[ymin:ymax, xmin:xmax]
        
        # Chuyển đổi sang thang độ xám
        prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)
        
        # Tính toán độ khác biệt
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Tính điểm số chuyển động
        motion_score = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1] * 255.0)
        
        return motion_score
    
    def update_prediction_history(self, behavior_type, probability):
        """
        Cập nhật lịch sử dự đoán cho một loại hành vi
        
        Args:
            behavior_type: Loại hành vi ('smoking', 'eating', 'drinking', 'calling')
            probability: Xác suất phát hiện
        """
        if behavior_type in self.prediction_history:
            self.prediction_history[behavior_type].append(probability)
    
    def detect_smoking(self, frame, prev_frame, face_landmarks):
        """
        Phát hiện hút thuốc dựa trên chuyển động tay lên miệng
        
        Args:
            frame: Khung hình hiện tại
            prev_frame: Khung hình trước đó
            face_landmarks: Các điểm đánh dấu trên khuôn mặt
            
        Returns:
            probability: Xác suất hút thuốc
        """
        # Trong triển khai thực tế, sẽ sử dụng mô hình CNN để phát hiện
        # Đây là một phiên bản đơn giản dựa trên chuyển động
        
        # Kiểm tra chuyển động trong vùng miệng
        mouth_motion = self._get_motion_in_region(prev_frame, frame, self.mouth_region, frame.shape)
        
        # Kiểm tra chuyển động trong các vùng tay
        hand_motions = [
            self._get_motion_in_region(prev_frame, frame, region, frame.shape)
            for region in self.hand_regions
        ]
        
        # Tính toán xác suất dựa trên chuyển động
        max_hand_motion = max(hand_motions)
        probability = 0.0
        
        # Nếu có chuyển động đáng kể ở cả tay và miệng
        if mouth_motion > 0.1 and max_hand_motion > 0.15:
            probability = min(1.0, (mouth_motion + max_hand_motion) / 2 * 2)
        
        self.update_prediction_history('smoking', probability)
        
        # Lấy xác suất trung bình từ lịch sử
        if len(self.prediction_history['smoking']) > 0:
            avg_probability = sum(self.prediction_history['smoking']) / len(self.prediction_history['smoking'])
            return avg_probability
        
        return probability
    
    def detect_eating(self, frame, prev_frame, face_landmarks):
        """
        Phát hiện ăn uống dựa trên chuyển động miệng và tay
        
        Args:
            frame: Khung hình hiện tại
            prev_frame: Khung hình trước đó
            face_landmarks: Các điểm đánh dấu trên khuôn mặt
            
        Returns:
            probability: Xác suất ăn uống
        """
        # Tương tự như phát hiện hút thuốc, nhưng với các mẫu chuyển động khác nhau
        mouth_motion = self._get_motion_in_region(prev_frame, frame, self.mouth_region, frame.shape)
        
        # Kiểm tra xem miệng có chuyển động mở thường xuyên không (ăn)
        probability = min(1.0, mouth_motion * 3)  # Trọng số cao hơn cho chuyển động miệng
        
        self.update_prediction_history('eating', probability)
        
        if len(self.prediction_history['eating']) > 0:
            avg_probability = sum(self.prediction_history['eating']) / len(self.prediction_history['eating'])
            return avg_probability
        
        return probability
    
    def detect_drinking(self, frame, prev_frame, face_landmarks):
        """
        Phát hiện uống nước dựa trên chuyển động đầu ngửa và tay
        
        Args:
            frame: Khung hình hiện tại
            prev_frame: Khung hình trước đó
            face_landmarks: Các điểm đánh dấu trên khuôn mặt
            
        Returns:
            probability: Xác suất uống nước
        """
        # Kiểm tra chuyển động trong vùng miệng
        mouth_motion = self._get_motion_in_region(prev_frame, frame, self.mouth_region, frame.shape)
        
        # Kiểm tra chuyển động trong các vùng tay
        hand_motions = [
            self._get_motion_in_region(prev_frame, frame, region, frame.shape)
            for region in self.hand_regions
        ]
        
        max_hand_motion = max(hand_motions)
        probability = 0.0
        
        # Nếu có chuyển động đáng kể ở cả tay và miệng
        if mouth_motion > 0.05 and max_hand_motion > 0.2:
            probability = min(1.0, (mouth_motion + max_hand_motion * 2) / 3)
        
        self.update_prediction_history('drinking', probability)
        
        if len(self.prediction_history['drinking']) > 0:
            avg_probability = sum(self.prediction_history['drinking']) / len(self.prediction_history['drinking'])
            return avg_probability
        
        return probability
    
    def detect_dangerous_behavior(self, frame, prev_frame, face_landmarks):
        """
        Phát hiện các hành vi nguy hiểm từ khung hình
        
        Args:
            frame: Khung hình hiện tại
            prev_frame: Khung hình trước đó
            face_landmarks: Các điểm đánh dấu trên khuôn mặt
            
        Returns:
            dangerous: Boolean chỉ ra có hành vi nguy hiểm hay không
            behavior_type: Loại hành vi nguy hiểm ('smoking', 'eating', 'drinking', 'calling')
            behavior_probability: Xác suất hành vi nguy hiểm
        """
        # Theo dõi thời gian
        current_time = time.time()
        
        # Phát hiện các hành vi khác nhau
        smoking_probability = self.detect_smoking(frame, prev_frame, face_landmarks)
        eating_probability = self.detect_eating(frame, prev_frame, face_landmarks)
        drinking_probability = self.detect_drinking(frame, prev_frame, face_landmarks)
        
        # Tìm hành vi có xác suất cao nhất
        behavior_probs = {
            "smoking": smoking_probability,
            "eating": eating_probability, 
            "drinking": drinking_probability
        }
        
        max_behavior = max(behavior_probs.items(), key=lambda x: x[1])
        behavior_type, behavior_probability = max_behavior
        
        # Nếu có hành vi vượt ngưỡng phát hiện
        if behavior_probability >= self.detection_threshold:
            # Nếu đây là sự bắt đầu của một hành vi mới
            if self.current_behavior_start is None or self.current_behavior_type != behavior_type:
                self.current_behavior_start = current_time
                self.current_behavior_type = behavior_type
            
            # Tính thời gian hành vi
            behavior_duration = current_time - self.current_behavior_start
            
            # Nếu đủ thời gian, báo cáo hành vi nguy hiểm
            if behavior_duration >= self.duration_threshold:
                return True, behavior_type, behavior_probability
        else:
            # Đặt lại nếu không phát hiện hành vi
            self.current_behavior_start = None
            self.current_behavior_type = None
        
        return False, None, 0.0 