import os
import sys
import time
from datetime import datetime
import threading
import uvicorn
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import json
import base64

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import DriverMonitoringSystem
from utils.alert_manager import AlertManager

app = FastAPI(
    title="Hệ thống Giám sát An Toàn Giao Thông Bằng AI",
    description="API cho hệ thống giám sát tài xế sử dụng AI",
    version="0.1.0"
)

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Các model dữ liệu
class AlertRequest(BaseModel):
    alert_type: str
    alert_level: float
    details: Optional[Dict[str, Any]] = None

class AlertResponse(BaseModel):
    success: bool
    message: str
    alert_id: Optional[str] = None

class AlertListResponse(BaseModel):
    alerts: List[Dict[str, Any]]
    count: int

class SystemStatusResponse(BaseModel):
    status: str
    camera_connected: bool
    monitoring_active: bool
    alerts_count: int
    calibrated: bool

# Biến toàn cục
monitoring_system = None
alert_manager = AlertManager()
system_thread = None
is_monitoring = False
last_frame = None
last_frame_lock = threading.Lock()

def start_monitoring_thread(camera_id: int = 0):
    """Bắt đầu giám sát trong một luồng riêng biệt"""
    global monitoring_system, is_monitoring, system_thread
    
    if is_monitoring:
        return False
    
    monitoring_system = DriverMonitoringSystem(camera_source=camera_id, show_ui=False)
    
    def run_monitoring():
        global is_monitoring, last_frame
        is_monitoring = True
        monitoring_system.start()
        is_monitoring = False
    
    system_thread = threading.Thread(target=run_monitoring)
    system_thread.daemon = True
    system_thread.start()
    
    return True

def stop_monitoring_thread():
    """Dừng giám sát"""
    global monitoring_system, is_monitoring
    
    if not is_monitoring or monitoring_system is None:
        return False
    
    monitoring_system.stop()
    is_monitoring = False
    
    return True

# Các route API
@app.get("/")
def read_root():
    """Kiểm tra API đang hoạt động"""
    return {
        "name": "Hệ thống Giám sát An Toàn Giao Thông Bằng AI",
        "version": "0.1.0",
        "status": "running"
    }

@app.post("/monitoring/start")
def start_monitoring(camera_id: int = 0):
    """Bắt đầu giám sát tài xế"""
    if is_monitoring:
        return {"success": False, "message": "Hệ thống đã đang giám sát"}
    
    success = start_monitoring_thread(camera_id)
    
    if success:
        return {"success": True, "message": "Bắt đầu giám sát thành công"}
    else:
        raise HTTPException(status_code=500, detail="Không thể bắt đầu giám sát")

@app.post("/monitoring/stop")
def stop_monitoring():
    """Dừng giám sát tài xế"""
    if not is_monitoring:
        return {"success": False, "message": "Hệ thống không đang giám sát"}
    
    success = stop_monitoring_thread()
    
    if success:
        return {"success": True, "message": "Dừng giám sát thành công"}
    else:
        raise HTTPException(status_code=500, detail="Không thể dừng giám sát")

@app.get("/monitoring/status")
def get_status():
    """Lấy trạng thái hiện tại của hệ thống"""
    global monitoring_system, is_monitoring
    
    camera_connected = False
    calibrated = False
    
    if monitoring_system is not None:
        camera_connected = monitoring_system.cap is not None and monitoring_system.cap.isOpened()
        # Giả định đã hiệu chuẩn nếu đang giám sát
        calibrated = is_monitoring
    
    # Lấy số lượng cảnh báo
    alerts_count = len(alert_manager.get_recent_alerts(100))
    
    return SystemStatusResponse(
        status="running" if is_monitoring else "stopped",
        camera_connected=camera_connected,
        monitoring_active=is_monitoring,
        alerts_count=alerts_count,
        calibrated=calibrated
    )

@app.get("/alerts/recent")
def get_recent_alerts(limit: int = 10):
    """Lấy danh sách cảnh báo gần đây"""
    alerts = alert_manager.get_recent_alerts(limit)
    return AlertListResponse(
        alerts=alerts,
        count=len(alerts)
    )

@app.get("/alerts/stats")
def get_alert_stats():
    """Lấy thống kê cảnh báo"""
    stats = alert_manager.get_alert_stats()
    return stats

@app.post("/alerts/trigger")
def trigger_alert(alert_request: AlertRequest):
    """Kích hoạt cảnh báo thủ công (cho mục đích thử nghiệm)"""
    success = alert_manager.trigger_alert(
        alert_request.alert_type, 
        alert_request.alert_level, 
        alert_request.details
    )
    
    if success:
        return AlertResponse(
            success=True,
            message=f"Đã kích hoạt cảnh báo {alert_request.alert_type}",
            alert_id=datetime.now().strftime("%Y%m%d%H%M%S")
        )
    else:
        return AlertResponse(
            success=False,
            message="Không thể kích hoạt cảnh báo"
        )

@app.get("/camera/frame")
def get_camera_frame():
    """Lấy khung hình hiện tại từ camera"""
    global monitoring_system, last_frame
    
    if not is_monitoring or monitoring_system is None:
        raise HTTPException(status_code=400, detail="Hệ thống không đang giám sát")
    
    with last_frame_lock:
        if monitoring_system.prev_frame is None:
            raise HTTPException(status_code=500, detail="Không có khung hình khả dụng")
        
        # Lấy khung hình đã xử lý
        frame = monitoring_system.add_info_to_frame(monitoring_system.prev_frame)
        
        # Chuyển đổi sang JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        
        # Mã hóa base64
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    
    # Trả về dữ liệu base64
    return {"image": jpg_as_text}

@app.post("/camera/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Phân tích hình ảnh tải lên"""
    global monitoring_system
    
    # Tạo hệ thống giám sát nếu chưa có
    if monitoring_system is None:
        monitoring_system = DriverMonitoringSystem(show_ui=False)
    
    try:
        # Đọc tệp hình ảnh
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Không thể đọc hình ảnh")
        
        # Xử lý hình ảnh
        processed_frame, face_landmarks, face_found = monitoring_system.face_detector.process_frame(image)
        
        # Kết quả phân tích
        analysis_result = {
            "face_detected": face_found,
            "drowsy": False,
            "distracted": False,
            "dangerous_behavior": False,
            "details": {}
        }
        
        # Nếu phát hiện khuôn mặt, tiến hành phân tích
        if face_found and face_landmarks:
            # Phát hiện buồn ngủ
            left_ear = monitoring_system.face_detector.calculate_eye_aspect_ratio(
                face_landmarks, monitoring_system.face_detector.LEFT_EYE)
            right_ear = monitoring_system.face_detector.calculate_eye_aspect_ratio(
                face_landmarks, monitoring_system.face_detector.RIGHT_EYE)
            mar = monitoring_system.face_detector.calculate_mouth_aspect_ratio(face_landmarks)
            roll, pitch, yaw = monitoring_system.face_detector.get_head_pose(face_landmarks, image.shape)
            
            # Phát hiện buồn ngủ
            drowsy, drowsy_level, drowsiness_indicators = monitoring_system.drowsiness_detector.detect_drowsiness(
                left_ear, right_ear, mar, pitch, yaw, roll)
            
            # Giả định đã hiệu chuẩn cho phát hiện mất tập trung
            monitoring_system.distraction_detector.reference_gaze = (0, 0, 0)
            monitoring_system.distraction_detector.calibrated = True
            
            # Phát hiện mất tập trung
            distracted, distraction_type, distraction_level = monitoring_system.distraction_detector.detect_distraction(
                yaw, pitch, roll, image, face_landmarks)
            
            # Cập nhật kết quả
            analysis_result["drowsy"] = drowsy
            analysis_result["distracted"] = distracted
            analysis_result["details"] = {
                "left_ear": round(left_ear, 3),
                "right_ear": round(right_ear, 3),
                "mar": round(mar, 3),
                "yaw": round(yaw, 3),
                "pitch": round(pitch, 3),
                "roll": round(roll, 3),
                "drowsy_level": round(drowsy_level, 3),
                "drowsiness_indicators": drowsiness_indicators
            }
            
            if distracted:
                analysis_result["details"]["distraction_type"] = distraction_type
                analysis_result["details"]["distraction_level"] = round(distraction_level, 3)
        
        # Thêm thông tin vào hình ảnh
        display_frame = monitoring_system.add_info_to_frame(processed_frame)
        
        # Chuyển đổi sang JPEG
        _, buffer = cv2.imencode('.jpg', display_frame)
        
        # Mã hóa base64
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        # Trả về cả kết quả và hình ảnh
        analysis_result["image"] = jpg_as_text
        
        return analysis_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý hình ảnh: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 