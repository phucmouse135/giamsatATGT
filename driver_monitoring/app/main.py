import os
import sys
import time
import cv2
import numpy as np
import threading
import argparse
from datetime import datetime

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.face_detector import FaceDetector
from utils.alert_manager import AlertManager
from models.drowsiness_detector import DrowsinessDetector
from models.distraction_detector import DistractionDetector
from models.dangerous_behavior_detector import DangerousBehaviorDetector

class DriverMonitoringSystem:
    """
    Hệ thống giám sát tài xế kết hợp các mô hình phát hiện khác nhau
    """
    def __init__(self, camera_source=0, show_ui=True, record=False, output_dir="output"):
        # Nguồn camera (0 là webcam mặc định)
        self.camera_source = camera_source
        
        # Có hiển thị giao diện người dùng hay không
        self.show_ui = show_ui
        
        # Có ghi lại video hay không
        self.record = record
        
        # Thư mục đầu ra
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Khởi tạo các thành phần
        self.face_detector = FaceDetector()
        self.drowsiness_detector = DrowsinessDetector()
        self.distraction_detector = DistractionDetector()
        self.dangerous_behavior_detector = DangerousBehaviorDetector()
        self.alert_manager = AlertManager(save_directory=os.path.join(output_dir, "alerts"))
        
        # Khởi tạo camera
        self.cap = None
        
        # Khung hình trước đó để phát hiện chuyển động
        self.prev_frame = None
        
        # Cờ dừng
        self.stop_flag = False
        
        # Thông tin hiển thị
        self.display_info = {
            "fps": 0,
            "face_detected": False,
            "drowsy": False,
            "distracted": False,
            "dangerous_behavior": False,
            "alerts": []
        }
        
        # Định dạng cho ghi video
        self.video_writer = None
    
    def start(self):
        """Bắt đầu giám sát"""
        try:
            # Mở camera
            self.cap = cv2.VideoCapture(self.camera_source)
            if not self.cap.isOpened():
                print(f"Không thể mở camera {self.camera_source}")
                return False
            
            # Kích thước khung hình
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Khởi tạo trình ghi video nếu cần
            if self.record:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_file = os.path.join(self.output_dir, f"recording_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(video_file, fourcc, 20.0, (frame_width, frame_height))
            
            print("Bắt đầu giám sát tài xế...")
            
            # Hiệu chuẩn cho phát hiện mất tập trung
            if self.show_ui:
                print("Vui lòng nhìn thẳng về phía trước để hiệu chuẩn hệ thống...")
            
            calibrated = False
            frame_count = 0
            start_time = time.time()
            
            while not self.stop_flag:
                # Đọc khung hình từ camera
                ret, frame = self.cap.read()
                if not ret:
                    print("Không thể đọc khung hình từ camera.")
                    break
                
                # Tính FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:
                    self.display_info["fps"] = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
                
                # Xử lý khung hình
                processed_frame, face_landmarks, face_found = self.face_detector.process_frame(frame)
                self.display_info["face_detected"] = face_found
                
                # Thêm thông tin vào khung hình
                display_frame = self.add_info_to_frame(processed_frame)
                
                # Hiển thị nếu yêu cầu
                if self.show_ui:
                    cv2.imshow("Driver Monitoring System", display_frame)
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC
                        break
                
                # Ghi video nếu yêu cầu
                if self.record and self.video_writer:
                    self.video_writer.write(display_frame)
                
                # Nếu phát hiện khuôn mặt, tiến hành phân tích
                if face_found and face_landmarks:
                    # Hiệu chuẩn phát hiện mất tập trung nếu cần
                    if not calibrated:
                        # Lấy góc hướng đầu
                        roll, pitch, yaw = self.face_detector.get_head_pose(face_landmarks, frame.shape)
                        
                        # Thêm dữ liệu hiệu chuẩn
                        calibrated = self.distraction_detector.calibrate(yaw, pitch, roll)
                        if calibrated and self.show_ui:
                            print("Hiệu chuẩn hoàn tất. Bắt đầu giám sát.")
                    else:
                        # Phát hiện buồn ngủ
                        left_ear = self.face_detector.calculate_eye_aspect_ratio(
                            face_landmarks, self.face_detector.LEFT_EYE)
                        right_ear = self.face_detector.calculate_eye_aspect_ratio(
                            face_landmarks, self.face_detector.RIGHT_EYE)
                        mar = self.face_detector.calculate_mouth_aspect_ratio(face_landmarks)
                        roll, pitch, yaw = self.face_detector.get_head_pose(face_landmarks, frame.shape)
                        
                        # Phát hiện buồn ngủ
                        drowsy, drowsy_level, drowsiness_indicators = self.drowsiness_detector.detect_drowsiness(
                            left_ear, right_ear, mar, pitch, yaw, roll)
                        self.display_info["drowsy"] = drowsy
                        
                        # Phát hiện mất tập trung
                        distracted, distraction_type, distraction_level = self.distraction_detector.detect_distraction(
                            yaw, pitch, roll, frame, face_landmarks)
                        self.display_info["distracted"] = distracted
                        
                        # Phát hiện hành vi nguy hiểm
                        if self.prev_frame is not None:
                            dangerous, behavior_type, behavior_level = self.dangerous_behavior_detector.detect_dangerous_behavior(
                                frame, self.prev_frame, face_landmarks)
                            self.display_info["dangerous_behavior"] = dangerous
                        
                        # Kích hoạt cảnh báo nếu cần
                        if drowsy:
                            # Tạo chi tiết cảnh báo
                            details = {
                                "indicators": ", ".join([k for k, v in drowsiness_indicators.items() if v]),
                                "left_ear": round(left_ear, 3),
                                "right_ear": round(right_ear, 3),
                                "mar": round(mar, 3)
                            }
                            
                            # Kích hoạt cảnh báo
                            self.alert_manager.trigger_alert("drowsiness", drowsy_level, details, frame)
                        
                        if distracted:
                            details = {
                                "type": distraction_type,
                                "yaw": round(yaw, 3),
                                "pitch": round(pitch, 3)
                            }
                            
                            self.alert_manager.trigger_alert("distraction", distraction_level, details, frame)
                        
                        if self.prev_frame is not None and dangerous:
                            details = {
                                "type": behavior_type,
                                "level": round(behavior_level, 3)
                            }
                            
                            self.alert_manager.trigger_alert("dangerous_behavior", behavior_level, details, frame)
                
                # Cập nhật khung hình trước đó
                self.prev_frame = frame.copy()
            
            # Dọn dẹp
            self._cleanup()
            
            return True
        
        except Exception as e:
            print(f"Lỗi: {str(e)}")
            self._cleanup()
            return False
    
    def stop(self):
        """Dừng giám sát"""
        self.stop_flag = True
    
    def _cleanup(self):
        """Dọn dẹp tài nguyên"""
        if self.cap:
            self.cap.release()
        
        if self.video_writer:
            self.video_writer.release()
        
        if self.show_ui:
            cv2.destroyAllWindows()
    
    def add_info_to_frame(self, frame):
        """
        Thêm thông tin giám sát vào khung hình
        
        Args:
            frame: Khung hình gốc
            
        Returns:
            display_frame: Khung hình với thông tin
        """
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Vẽ nền bán trong suốt cho thông tin
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        # Thêm thông tin FPS
        cv2.putText(display_frame, f"FPS: {self.display_info['fps']:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Thêm thông tin phát hiện khuôn mặt
        face_color = (0, 255, 0) if self.display_info["face_detected"] else (0, 0, 255)
        cv2.putText(display_frame, f"Khuôn mặt: {'Có' if self.display_info['face_detected'] else 'Không'}", 
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
        
        # Thêm thông tin trạng thái
        drowsy_color = (0, 0, 255) if self.display_info["drowsy"] else (0, 255, 0)
        distracted_color = (0, 0, 255) if self.display_info["distracted"] else (0, 255, 0)
        dangerous_color = (0, 0, 255) if self.display_info["dangerous_behavior"] else (0, 255, 0)
        
        cv2.putText(display_frame, f"Buồn ngủ: {'Có' if self.display_info['drowsy'] else 'Không'}", 
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, drowsy_color, 2)
        
        cv2.putText(display_frame, f"Mất tập trung: {'Có' if self.display_info['distracted'] else 'Không'}", 
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, distracted_color, 2)
        
        cv2.putText(display_frame, f"Hành vi nguy hiểm: {'Có' if self.display_info['dangerous_behavior'] else 'Không'}", 
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dangerous_color, 2)
        
        return display_frame

def parse_arguments():
    """Phân tích các đối số dòng lệnh"""
    parser = argparse.ArgumentParser(description="Hệ thống giám sát tài xế bằng AI")
    parser.add_argument("--camera", type=int, default=0, help="ID của camera (mặc định: 0)")
    parser.add_argument("--no-ui", action="store_true", help="Không hiển thị giao diện người dùng")
    parser.add_argument("--record", action="store_true", help="Ghi lại video")
    parser.add_argument("--output", type=str, default="output", help="Thư mục đầu ra (mặc định: output)")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Phân tích đối số
    args = parse_arguments()
    
    # Khởi tạo hệ thống
    system = DriverMonitoringSystem(
        camera_source=args.camera,
        show_ui=not args.no_ui,
        record=args.record,
        output_dir=args.output
    )
    
    # Bắt đầu giám sát
    system.start() 