import time
import threading
import json
import os
from datetime import datetime
import cv2
import numpy as np

class AlertManager:
    """
    Quản lý cảnh báo và thông báo cho người lái xe và quản lý
    """
    def __init__(self, alert_threshold=0.6, cooldown_period=5.0, save_directory="alerts"):
        # Ngưỡng cảnh báo (0-1)
        self.alert_threshold = alert_threshold
        
        # Thời gian chờ giữa các cảnh báo (giây)
        self.cooldown_period = cooldown_period
        
        # Thời gian cảnh báo cuối cùng
        self.last_alert_time = {
            "drowsiness": 0,
            "distraction": 0,
            "dangerous_behavior": 0
        }
        
        # Lịch sử cảnh báo
        self.alert_history = []
        
        # Thời gian bắt đầu cảnh báo liên tục
        self.continuous_alert_start = {
            "drowsiness": None,
            "distraction": None, 
            "dangerous_behavior": None
        }
        
        # Thư mục lưu cảnh báo
        self.save_directory = save_directory
        os.makedirs(save_directory, exist_ok=True)
        
        # Khóa đồng bộ
        self.lock = threading.Lock()
        
        # Đường dẫn tệp âm thanh
        self.alert_sounds = {
            "drowsiness": "sounds/drowsiness_alert.wav",
            "distraction": "sounds/distraction_alert.wav",
            "dangerous_behavior": "sounds/dangerous_behavior_alert.wav",
            "general": "sounds/general_alert.wav"
        }
    
    def can_alert(self, alert_type):
        """
        Kiểm tra xem có thể đưa ra cảnh báo hay không dựa trên thời gian chờ
        
        Args:
            alert_type: Loại cảnh báo
            
        Returns:
            can_alert: Boolean chỉ ra có thể cảnh báo hay không
        """
        current_time = time.time()
        with self.lock:
            elapsed_time = current_time - self.last_alert_time.get(alert_type, 0)
            return elapsed_time >= self.cooldown_period
    
    def trigger_alert(self, alert_type, alert_level, details=None, frame=None):
        """
        Kích hoạt cảnh báo nếu vượt quá ngưỡng
        
        Args:
            alert_type: Loại cảnh báo ('drowsiness', 'distraction', 'dangerous_behavior')
            alert_level: Mức độ cảnh báo (0-1)
            details: Chi tiết bổ sung về cảnh báo
            frame: Khung hình hiện tại (tùy chọn)
            
        Returns:
            alerted: Boolean chỉ ra cảnh báo đã được kích hoạt hay không
        """
        # Chỉ cảnh báo nếu vượt quá ngưỡng
        if alert_level < self.alert_threshold:
            self.continuous_alert_start[alert_type] = None
            return False
        
        current_time = time.time()
        
        # Ghi lại thời gian bắt đầu cảnh báo liên tục
        if self.continuous_alert_start[alert_type] is None:
            self.continuous_alert_start[alert_type] = current_time
        
        # Thời gian cảnh báo liên tục
        continuous_duration = current_time - self.continuous_alert_start[alert_type]
        
        # Tăng mức độ cảnh báo dựa trên thời gian liên tục
        adjusted_level = min(1.0, alert_level * (1.0 + continuous_duration / 10.0))
        
        # Kiểm tra xem có thể cảnh báo hay không
        if self.can_alert(alert_type):
            with self.lock:
                # Cập nhật thời gian cảnh báo cuối cùng
                self.last_alert_time[alert_type] = current_time
                
                # Tạo cảnh báo
                alert_data = {
                    "type": alert_type,
                    "level": adjusted_level,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "details": details or {}
                }
                
                # Thêm vào lịch sử
                self.alert_history.append(alert_data)
                
                # Giới hạn lịch sử
                if len(self.alert_history) > 100:
                    self.alert_history.pop(0)
                
                # Lưu khung hình nếu có
                if frame is not None:
                    self._save_alert_image(frame, alert_data)
                
                # Phát âm thanh cảnh báo
                self._play_alert_sound(alert_type)
                
                # Gửi thông báo đến quản lý phương tiện
                self._send_notification(alert_data)
                
                return True
        
        return False
    
    def _save_alert_image(self, frame, alert_data):
        """
        Lưu hình ảnh khi có cảnh báo
        
        Args:
            frame: Khung hình hiện tại
            alert_data: Dữ liệu cảnh báo
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alert_type = alert_data["type"]
        filename = f"{self.save_directory}/{alert_type}_{timestamp}.jpg"
        
        # Thêm văn bản cảnh báo vào hình ảnh
        alert_frame = frame.copy()
        alert_text = f"{alert_type.upper()} - Level: {alert_data['level']:.2f}"
        details = alert_data.get("details", {})
        
        cv2.putText(alert_frame, alert_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        y_offset = 60
        for key, value in details.items():
            detail_text = f"{key}: {value}"
            cv2.putText(alert_frame, detail_text, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            y_offset += 25
        
        cv2.imwrite(filename, alert_frame)
    
    def _play_alert_sound(self, alert_type):
        """
        Phát âm thanh cảnh báo
        
        Args:
            alert_type: Loại cảnh báo
        """
        # Trong triển khai thực tế, sẽ sử dụng thư viện âm thanh như pygame
        sound_file = self.alert_sounds.get(alert_type, self.alert_sounds["general"])
        print(f"[ALERT SOUND] Playing {sound_file}")
    
    def _send_notification(self, alert_data):
        """
        Gửi thông báo đến quản lý phương tiện
        
        Args:
            alert_data: Dữ liệu cảnh báo
        """
        # Trong triển khai thực tế, sẽ gửi thông báo qua API, SMS, email, v.v.
        print(f"[NOTIFICATION] {json.dumps(alert_data)}")
    
    def get_recent_alerts(self, limit=10):
        """
        Lấy danh sách cảnh báo gần đây
        
        Args:
            limit: Số lượng cảnh báo tối đa để trả về
            
        Returns:
            recent_alerts: Danh sách cảnh báo gần đây
        """
        with self.lock:
            return self.alert_history[-limit:] if self.alert_history else []
    
    def get_alert_stats(self):
        """
        Lấy thống kê cảnh báo
        
        Returns:
            stats: Thống kê cảnh báo theo loại
        """
        with self.lock:
            stats = {
                "drowsiness": 0,
                "distraction": 0,
                "dangerous_behavior": 0,
                "total": len(self.alert_history)
            }
            
            for alert in self.alert_history:
                alert_type = alert.get("type")
                if alert_type in stats:
                    stats[alert_type] += 1
            
            return stats 