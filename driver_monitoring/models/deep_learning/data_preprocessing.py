import cv2
import numpy as np
from collections import deque
import tensorflow as tf
import mediapipe as mp
from typing import List, Tuple, Dict, Optional, Union

class FaceDataPreprocessor:
    """
    Lớp tiền xử lý dữ liệu khuôn mặt cho mô hình deep learning
    """
    def __init__(self, 
                 face_img_size: Tuple[int, int] = (128, 128),
                 sequence_length: int = 20,
                 use_landmarks: bool = True,
                 normalize: bool = True):
        """
        Khởi tạo bộ tiền xử lý dữ liệu
        
        Args:
            face_img_size: Kích thước ảnh khuôn mặt đầu vào cho CNN
            sequence_length: Số khung hình liên tiếp cho LSTM
            use_landmarks: Có sử dụng các điểm đánh dấu khuôn mặt hay không
            normalize: Có chuẩn hóa dữ liệu hay không
        """
        self.face_img_size = face_img_size
        self.sequence_length = sequence_length
        self.use_landmarks = use_landmarks
        self.normalize = normalize
        
        # Lưu trữ chuỗi dữ liệu cho LSTM
        self.face_sequence = deque(maxlen=sequence_length)
        self.eye_sequence = deque(maxlen=sequence_length)
        self.mouth_sequence = deque(maxlen=sequence_length)
        self.ear_sequence = deque(maxlen=sequence_length)
        self.mar_sequence = deque(maxlen=sequence_length)
        self.head_pose_sequence = deque(maxlen=sequence_length)
        
        # Định nghĩa quan điểm landmark
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Điểm landmark cho mắt
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # Điểm landmark cho miệng
        self.LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
        
    def extract_face_roi(self, frame, face_landmarks) -> np.ndarray:
        """
        Trích xuất vùng khuôn mặt từ khung hình
        
        Args:
            frame: Khung hình đầu vào
            face_landmarks: Các điểm đánh dấu khuôn mặt
            
        Returns:
            face_roi: Vùng ảnh khuôn mặt đã được cắt và định kích thước lại
        """
        h, w = frame.shape[:2]
        
        # Lấy tọa độ tất cả các điểm đánh dấu
        points = []
        for landmark in face_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            points.append((x, y))
        
        points = np.array(points)
        
        # Tìm hộp giới hạn cho khuôn mặt
        x, y, width, height = cv2.boundingRect(points)
        
        # Mở rộng hộp giới hạn để bao gồm toàn bộ khuôn mặt
        padding_x = int(width * 0.2)
        padding_y = int(height * 0.2)
        
        # Đảm bảo tọa độ hợp lệ
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(w, x + width + padding_x)
        y2 = min(h, y + height + padding_y)
        
        # Cắt khuôn mặt
        face_roi = frame[y1:y2, x1:x2]
        
        # Đổi kích thước về kích thước đầu vào của mô hình
        face_roi = cv2.resize(face_roi, self.face_img_size)
        
        # Chuẩn hóa hình ảnh nếu cần
        if self.normalize:
            face_roi = face_roi.astype(np.float32) / 255.0
        
        return face_roi
    
    def extract_eye_rois(self, frame, face_landmarks) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trích xuất vùng mắt trái và phải
        
        Args:
            frame: Khung hình đầu vào
            face_landmarks: Các điểm đánh dấu khuôn mặt
            
        Returns:
            left_eye_roi, right_eye_roi: Vùng ảnh mắt trái và phải
        """
        h, w = frame.shape[:2]
        
        # Trích xuất mắt trái
        left_eye_points = []
        for idx in self.LEFT_EYE:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            left_eye_points.append((x, y))
        
        left_eye_points = np.array(left_eye_points)
        l_x, l_y, l_w, l_h = cv2.boundingRect(left_eye_points)
        
        # Mở rộng vùng mắt
        l_padding_x = int(l_w * 0.4)
        l_padding_y = int(l_h * 0.4)
        l_x1 = max(0, l_x - l_padding_x)
        l_y1 = max(0, l_y - l_padding_y)
        l_x2 = min(w, l_x + l_w + l_padding_x)
        l_y2 = min(h, l_y + l_h + l_padding_y)
        
        left_eye_roi = frame[l_y1:l_y2, l_x1:l_x2]
        
        # Trích xuất mắt phải
        right_eye_points = []
        for idx in self.RIGHT_EYE:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            right_eye_points.append((x, y))
        
        right_eye_points = np.array(right_eye_points)
        r_x, r_y, r_w, r_h = cv2.boundingRect(right_eye_points)
        
        # Mở rộng vùng mắt
        r_padding_x = int(r_w * 0.4)
        r_padding_y = int(r_h * 0.4)
        r_x1 = max(0, r_x - r_padding_x)
        r_y1 = max(0, r_y - r_padding_y)
        r_x2 = min(w, r_x + r_w + r_padding_x)
        r_y2 = min(h, r_y + r_h + r_padding_y)
        
        right_eye_roi = frame[r_y1:r_y2, r_x1:r_x2]
        
        # Đổi kích thước
        eye_size = (64, 32)  # Kích thước phù hợp cho mắt
        if left_eye_roi.size > 0:
            left_eye_roi = cv2.resize(left_eye_roi, eye_size)
        else:
            left_eye_roi = np.zeros((eye_size[1], eye_size[0], 3), dtype=np.uint8)
            
        if right_eye_roi.size > 0:
            right_eye_roi = cv2.resize(right_eye_roi, eye_size)
        else:
            right_eye_roi = np.zeros((eye_size[1], eye_size[0], 3), dtype=np.uint8)
        
        # Chuẩn hóa
        if self.normalize:
            left_eye_roi = left_eye_roi.astype(np.float32) / 255.0
            right_eye_roi = right_eye_roi.astype(np.float32) / 255.0
        
        return left_eye_roi, right_eye_roi
    
    def extract_mouth_roi(self, frame, face_landmarks) -> np.ndarray:
        """
        Trích xuất vùng miệng
        
        Args:
            frame: Khung hình đầu vào
            face_landmarks: Các điểm đánh dấu khuôn mặt
            
        Returns:
            mouth_roi: Vùng ảnh miệng
        """
        h, w = frame.shape[:2]
        
        # Trích xuất miệng
        mouth_points = []
        for idx in self.LIPS:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            mouth_points.append((x, y))
        
        mouth_points = np.array(mouth_points)
        m_x, m_y, m_w, m_h = cv2.boundingRect(mouth_points)
        
        # Mở rộng vùng miệng
        m_padding_x = int(m_w * 0.3)
        m_padding_y = int(m_h * 0.3)
        m_x1 = max(0, m_x - m_padding_x)
        m_y1 = max(0, m_y - m_padding_y)
        m_x2 = min(w, m_x + m_w + m_padding_x)
        m_y2 = min(h, m_y + m_h + m_padding_y)
        
        mouth_roi = frame[m_y1:m_y2, m_x1:m_x2]
        
        # Đổi kích thước
        mouth_size = (64, 32)  # Kích thước phù hợp cho miệng
        if mouth_roi.size > 0:
            mouth_roi = cv2.resize(mouth_roi, mouth_size)
        else:
            mouth_roi = np.zeros((mouth_size[1], mouth_size[0], 3), dtype=np.uint8)
        
        # Chuẩn hóa
        if self.normalize:
            mouth_roi = mouth_roi.astype(np.float32) / 255.0
        
        return mouth_roi
    
    def extract_landmarks_features(self, face_landmarks) -> np.ndarray:
        """
        Trích xuất các đặc trưng từ các điểm đánh dấu khuôn mặt
        
        Args:
            face_landmarks: Các điểm đánh dấu khuôn mặt
            
        Returns:
            features: Mảng các đặc trưng từ các điểm đánh dấu
        """
        if not face_landmarks:
            return np.zeros((1, 468, 2), dtype=np.float32)
        
        # Trích xuất tất cả các điểm đánh dấu
        points = []
        for landmark in face_landmarks.landmark:
            points.append([landmark.x, landmark.y])
        
        # Chuyển thành mảng numpy
        features = np.array(points, dtype=np.float32)
        
        # Đảm bảo kích thước đúng
        if features.shape[0] < 468:
            # Đệm với giá trị 0 nếu thiếu điểm đánh dấu
            pad_length = 468 - features.shape[0]
            padding = np.zeros((pad_length, 2), dtype=np.float32)
            features = np.vstack([features, padding])
        
        # Thêm chiều batch
        features = np.expand_dims(features, axis=0)
        
        return features
    
    def update_sequences(self, face_roi, left_eye_roi, right_eye_roi, mouth_roi, ear, mar, head_pose):
        """
        Cập nhật các chuỗi thời gian cho LSTM
        
        Args:
            face_roi: Vùng ảnh khuôn mặt
            left_eye_roi, right_eye_roi: Vùng ảnh mắt trái và phải
            mouth_roi: Vùng ảnh miệng
            ear: Tỷ lệ khía cạnh mắt
            mar: Tỷ lệ khía cạnh miệng
            head_pose: Góc nghiêng, ngẩng và quay đầu (roll, pitch, yaw)
        """
        # Cập nhật chuỗi
        self.face_sequence.append(face_roi)
        
        # Ghép mắt trái và phải thành một ảnh
        eye_combined = np.hstack((left_eye_roi, right_eye_roi))
        self.eye_sequence.append(eye_combined)
        
        self.mouth_sequence.append(mouth_roi)
        self.ear_sequence.append(ear)
        self.mar_sequence.append(mar)
        self.head_pose_sequence.append(head_pose)
    
    def get_sequence_data(self):
        """
        Lấy dữ liệu chuỗi thời gian cho LSTM
        
        Returns:
            sequence_data: Từ điển chứa các chuỗi dữ liệu
        """
        # Chuyển đổi deque thành mảng numpy
        face_seq = np.array(list(self.face_sequence))
        eye_seq = np.array(list(self.eye_sequence))
        mouth_seq = np.array(list(self.mouth_sequence))
        
        # Đảm bảo đủ độ dài chuỗi
        if len(face_seq) < self.sequence_length:
            # Đệm với khung hình đầu tiên
            padding_length = self.sequence_length - len(face_seq)
            if len(face_seq) > 0:
                face_padding = np.repeat(np.expand_dims(face_seq[0], axis=0), padding_length, axis=0)
                eye_padding = np.repeat(np.expand_dims(eye_seq[0], axis=0), padding_length, axis=0)
                mouth_padding = np.repeat(np.expand_dims(mouth_seq[0], axis=0), padding_length, axis=0)
            else:
                # Nếu chuỗi trống, đệm với 0
                face_padding = np.zeros((padding_length, *self.face_img_size, 3), dtype=np.float32)
                eye_padding = np.zeros((padding_length, 32, 128, 3), dtype=np.float32)  # 2 mắt ghép lại
                mouth_padding = np.zeros((padding_length, 32, 64, 3), dtype=np.float32)
            
            face_seq = np.vstack([face_padding, face_seq])
            eye_seq = np.vstack([eye_padding, eye_seq])
            mouth_seq = np.vstack([mouth_padding, mouth_seq])
        
        # Đảm bảo kích thước đúng cho mô hình
        face_seq = face_seq.astype(np.float32)
        eye_seq = eye_seq.astype(np.float32)
        mouth_seq = mouth_seq.astype(np.float32)
        
        # Chuẩn bị dữ liệu số
        ear_seq = np.array(list(self.ear_sequence), dtype=np.float32)
        mar_seq = np.array(list(self.mar_sequence), dtype=np.float32)
        head_pose_seq = np.array(list(self.head_pose_sequence), dtype=np.float32)
        
        # Đệm dữ liệu số nếu cần
        if len(ear_seq) < self.sequence_length:
            padding_length = self.sequence_length - len(ear_seq)
            
            if len(ear_seq) > 0:
                ear_padding = np.repeat(ear_seq[0], padding_length)
                mar_padding = np.repeat(mar_seq[0], padding_length)
                head_pose_padding = np.repeat(np.expand_dims(head_pose_seq[0], axis=0), padding_length, axis=0)
            else:
                ear_padding = np.zeros(padding_length, dtype=np.float32)
                mar_padding = np.zeros(padding_length, dtype=np.float32)
                head_pose_padding = np.zeros((padding_length, 3), dtype=np.float32)
            
            ear_seq = np.concatenate([ear_padding, ear_seq])
            mar_seq = np.concatenate([mar_padding, mar_seq])
            head_pose_seq = np.vstack([head_pose_padding, head_pose_seq])
        
        # Tạo vector đặc trưng kết hợp
        numeric_features = np.column_stack([ear_seq, mar_seq, head_pose_seq])
        
        return {
            'face_sequence': face_seq,
            'eye_sequence': eye_seq,
            'mouth_sequence': mouth_seq,
            'numeric_features': numeric_features
        }
    
    def process_frame(self, frame, face_landmarks, left_ear, right_ear, mar, roll, pitch, yaw):
        """
        Xử lý một khung hình và trả về dữ liệu cho các mô hình deep learning
        
        Args:
            frame: Khung hình đầu vào
            face_landmarks: Các điểm đánh dấu khuôn mặt
            left_ear, right_ear: Tỷ lệ khía cạnh mắt trái và phải
            mar: Tỷ lệ khía cạnh miệng
            roll, pitch, yaw: Góc nghiêng, ngẩng và quay đầu
            
        Returns:
            processed_data: Từ điển chứa dữ liệu đã xử lý
        """
        # Tính toán tỷ lệ khía cạnh mắt trung bình
        ear = (left_ear + right_ear) / 2.0
        
        # Trích xuất vùng ảnh
        face_roi = self.extract_face_roi(frame, face_landmarks)
        left_eye_roi, right_eye_roi = self.extract_eye_rois(frame, face_landmarks)
        mouth_roi = self.extract_mouth_roi(frame, face_landmarks)
        
        # Cập nhật chuỗi thời gian
        head_pose = (roll, pitch, yaw)
        self.update_sequences(face_roi, left_eye_roi, right_eye_roi, mouth_roi, ear, mar, head_pose)
        
        # Lấy dữ liệu chuỗi cho LSTM
        sequence_data = self.get_sequence_data()
        
        # Trích xuất đặc trưng từ các điểm đánh dấu nếu cần
        landmarks_features = None
        if self.use_landmarks:
            landmarks_features = self.extract_landmarks_features(face_landmarks)
        
        # Tạo dữ liệu đầu vào cho mô hình
        processed_data = {
            'face_image': np.expand_dims(face_roi, axis=0),  # Thêm chiều batch
            'eye_image': np.expand_dims(np.hstack((left_eye_roi, right_eye_roi)), axis=0),
            'mouth_image': np.expand_dims(mouth_roi, axis=0),
            'landmarks': landmarks_features,
            'ear': ear,
            'mar': mar,
            'head_pose': head_pose,
            'sequences': sequence_data
        }
        
        return processed_data 