import cv2
import mediapipe as mp
import numpy as np

class FaceDetector:
    """
    Lớp phát hiện và phân tích khuôn mặt sử dụng MediaPipe
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Khởi tạo bộ phát hiện khuôn mặt
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Điểm landmark cho mắt
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # Điểm landmark cho miệng
        self.LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
        
        # Điểm landmark cho hướng đầu
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
    def process_frame(self, frame):
        """
        Xử lý khung hình để phát hiện và phân tích khuôn mặt
        
        Args:
            frame: Khung hình từ camera
            
        Returns:
            processed_frame: Khung hình đã được xử lý với các điểm đánh dấu
            face_landmarks: Các điểm đánh dấu trên khuôn mặt nếu phát hiện được
            face_found: Boolean cho biết có phát hiện khuôn mặt hay không
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        processed_frame = frame.copy()
        face_found = False
        face_landmarks = None
        
        if results.multi_face_landmarks:
            face_found = True
            face_landmarks = results.multi_face_landmarks[0]
            
            # Vẽ các điểm đánh dấu lên khuôn mặt
            self.mp_drawing.draw_landmarks(
                image=processed_frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            # Vẽ đường viền khuôn mặt
            self.mp_drawing.draw_landmarks(
                image=processed_frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
            
            # Vẽ mống mắt
            self.mp_drawing.draw_landmarks(
                image=processed_frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style())
            
        return processed_frame, face_landmarks, face_found
    
    def calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        """
        Tính tỷ lệ khía cạnh mắt (EAR) để phát hiện chớp mắt
        
        Args:
            landmarks: Các điểm đánh dấu trên khuôn mặt
            eye_indices: Chỉ số của các điểm đánh dấu trên mắt
            
        Returns:
            ear: Tỷ lệ khía cạnh mắt
        """
        points = []
        for index in eye_indices:
            landmark = landmarks.landmark[index]
            points.append([landmark.x, landmark.y])
        
        points = np.array(points)
        
        # Tính toán EAR theo công thức
        A = np.linalg.norm(points[1] - points[5])
        B = np.linalg.norm(points[2] - points[4])
        C = np.linalg.norm(points[0] - points[3])
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_mouth_aspect_ratio(self, landmarks):
        """
        Tính tỷ lệ khía cạnh miệng để phát hiện ngáp
        
        Args:
            landmarks: Các điểm đánh dấu trên khuôn mặt
            
        Returns:
            mar: Tỷ lệ khía cạnh miệng
        """
        points = []
        for index in self.LIPS:
            landmark = landmarks.landmark[index]
            points.append([landmark.x, landmark.y])
        
        points = np.array(points)
        
        # Tính toán phạm vi mở miệng
        height = np.linalg.norm(points[2] - points[6])
        width = np.linalg.norm(points[0] - points[4])
        
        mar = height / width
        return mar
    
    def get_head_pose(self, landmarks, frame_shape):
        """
        Ước tính hướng đầu từ các điểm đánh dấu trên khuôn mặt
        
        Args:
            landmarks: Các điểm đánh dấu trên khuôn mặt
            frame_shape: Kích thước khung hình
            
        Returns:
            roll, pitch, yaw: Góc nghiêng, ngẩng và quay đầu
        """
        # Tọa độ hình ảnh 2D
        image_points = []
        model_points = []
        
        # Chọn các điểm đánh dấu để ước tính hướng đầu
        indices = [33, 263, 1, 61, 291, 199]
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Đỉnh mũi
            (0.0, -330.0, -65.0),        # Cằm
            (-225.0, 170.0, -135.0),     # Góc trái miệng
            (225.0, 170.0, -135.0),      # Góc phải miệng
            (-150.0, -150.0, -125.0),    # Góc trái mắt
            (150.0, -150.0, -125.0)      # Góc phải mắt
        ])
        
        for idx in indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * frame_shape[1])
            y = int(landmark.y * frame_shape[0])
            image_points.append((x, y))
        
        image_points = np.array(image_points, dtype="double")
        
        # Tham số camera
        focal_length = frame_shape[1]
        center = (frame_shape[1]/2, frame_shape[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        
        dist_coeffs = np.zeros((4, 1))
        
        # Giải bài toán PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        # Chuyển đổi vector quay thành ma trận quay và sau đó là góc Euler
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
        
        pitch, yaw, roll = [np.radians(angle) for angle in euler_angles]
        
        return roll, pitch, yaw 