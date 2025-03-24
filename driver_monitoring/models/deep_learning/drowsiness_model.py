import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import LSTM, TimeDistributed, Concatenate, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from driver_monitoring.models.deep_learning import (
    FaceDataPreprocessor, 
    DrowsinessDetectorModel,
    DeepLearningModelManager
)


class DrowsinessDetectorModel:
    """
    Mô hình CNN-LSTM cho phát hiện buồn ngủ sử dụng nhiều nguồn dữ liệu
    """
    def __init__(self, 
                 face_shape: Tuple[int, int] = (128, 128),
                 eye_shape: Tuple[int, int] = (32, 128),
                 mouth_shape: Tuple[int, int] = (32, 64),
                 sequence_length: int = 20,
                 use_pretrained: bool = True,
                 use_landmarks: bool = True,
                 dropout_rate: float = 0.5,
                 l2_reg: float = 0.001):
        """
        Khởi tạo mô hình
        
        Args:
            face_shape: Kích thước ảnh khuôn mặt đầu vào (height, width)
            eye_shape: Kích thước ảnh mắt đầu vào (height, width)
            mouth_shape: Kích thước ảnh miệng đầu vào (height, width)
            sequence_length: Số khung hình liên tiếp
            use_pretrained: Có sử dụng mô hình pretrained hay không
            use_landmarks: Có sử dụng đặc trưng từ các điểm đánh dấu hay không
            dropout_rate: Tỷ lệ dropout
            l2_reg: Hằng số chính quy L2
        """
        self.face_shape = face_shape
        self.eye_shape = eye_shape
        self.mouth_shape = mouth_shape
        self.sequence_length = sequence_length
        self.use_pretrained = use_pretrained
        self.use_landmarks = use_landmarks
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        # Xây dựng mô hình
        self.model = self._build_model()
        
    def _build_face_feature_extractor(self) -> Model:
        """
        Xây dựng phần trích xuất đặc trưng từ khuôn mặt
        
        Returns:
            model: Mô hình trích xuất đặc trưng khuôn mặt
        """
        if self.use_pretrained:
            # Sử dụng MobileNetV2 làm base model
            base_model = MobileNetV2(
                input_shape=(*self.face_shape, 3),
                include_top=False,
                weights='imagenet'
            )
            
            # Đóng băng các lớp pretrained
            for layer in base_model.layers:
                layer.trainable = False
                
            face_input = base_model.input
            x = base_model.output
            
        else:
            # Xây dựng CNN từ đầu
            face_input = Input(shape=(*self.face_shape, 3))
            
            x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(self.l2_reg))(face_input)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            
            x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(self.l2_reg))(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            
            x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(self.l2_reg))(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            
            x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(self.l2_reg))(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Tiếp tục xử lý sau khi trích xuất đặc trưng
        x = Flatten()(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(self.l2_reg))(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(self.l2_reg))(x)
        
        face_features = Dense(128, activation='relu', name='face_features')(x)
        
        return Model(inputs=face_input, outputs=face_features, name='face_feature_extractor')
    
    def _build_eye_feature_extractor(self) -> Model:
        """
        Xây dựng phần trích xuất đặc trưng từ mắt
        
        Returns:
            model: Mô hình trích xuất đặc trưng mắt
        """
        eye_input = Input(shape=(*self.eye_shape, 3))
        
        x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(self.l2_reg))(eye_input)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Flatten()(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(self.l2_reg))(x)
        x = Dropout(self.dropout_rate)(x)
        
        eye_features = Dense(64, activation='relu', name='eye_features')(x)
        
        return Model(inputs=eye_input, outputs=eye_features, name='eye_feature_extractor')
    
    def _build_mouth_feature_extractor(self) -> Model:
        """
        Xây dựng phần trích xuất đặc trưng từ miệng
        
        Returns:
            model: Mô hình trích xuất đặc trưng miệng
        """
        mouth_input = Input(shape=(*self.mouth_shape, 3))
        
        x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(self.l2_reg))(mouth_input)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Flatten()(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(self.l2_reg))(x)
        x = Dropout(self.dropout_rate)(x)
        
        mouth_features = Dense(32, activation='relu', name='mouth_features')(x)
        
        return Model(inputs=mouth_input, outputs=mouth_features, name='mouth_feature_extractor')
    
    def _build_landmarks_processor(self) -> Model:
        """
        Xây dựng phần xử lý đặc trưng từ các điểm đánh dấu
        
        Returns:
            model: Mô hình xử lý các điểm đánh dấu
        """
        landmarks_input = Input(shape=(468, 2))  # 468 điểm đánh dấu, mỗi điểm có 2 tọa độ (x, y)
        
        x = Flatten()(landmarks_input)
        x = Dense(512, activation='relu', kernel_regularizer=l2(self.l2_reg))(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(self.l2_reg))(x)
        x = Dropout(self.dropout_rate)(x)
        
        landmarks_features = Dense(64, activation='relu', name='landmarks_features')(x)
        
        return Model(inputs=landmarks_input, outputs=landmarks_features, name='landmarks_processor')
    
    def _build_temporal_model(self) -> Model:
        """
        Xây dựng mô hình xử lý dữ liệu theo chuỗi thời gian (LSTM)
        
        Returns:
            model: Mô hình LSTM
        """
        # Đầu vào cho face sequence
        face_sequence_input = Input(shape=(self.sequence_length, *self.face_shape, 3))
        
        # Đầu vào cho eye sequence
        eye_sequence_input = Input(shape=(self.sequence_length, *self.eye_shape, 3))
        
        # Đầu vào cho mouth sequence
        mouth_sequence_input = Input(shape=(self.sequence_length, *self.mouth_shape, 3))
        
        # Đầu vào cho numeric features (EAR, MAR, head pose)
        numeric_input = Input(shape=(self.sequence_length, 5))  # EAR, MAR, roll, pitch, yaw
        
        # Trích xuất đặc trưng từ face sequence
        face_extractor = self._build_face_feature_extractor()
        face_td = TimeDistributed(face_extractor)(face_sequence_input)
        
        # Trích xuất đặc trưng từ eye sequence
        eye_extractor = self._build_eye_feature_extractor()
        eye_td = TimeDistributed(eye_extractor)(eye_sequence_input)
        
        # Trích xuất đặc trưng từ mouth sequence
        mouth_extractor = self._build_mouth_feature_extractor()
        mouth_td = TimeDistributed(mouth_extractor)(mouth_sequence_input)
        
        # Kết hợp các đặc trưng
        combined_features = Concatenate()([face_td, eye_td, mouth_td, numeric_input])
        
        # LSTM layers
        x = LSTM(256, return_sequences=True)(combined_features)
        x = Dropout(self.dropout_rate)(x)
        x = LSTM(128)(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Fully connected layers
        x = Dense(64, activation='relu', kernel_regularizer=l2(self.l2_reg))(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Lớp đầu ra
        output = Dense(1, activation='sigmoid', name='drowsiness')(x)
        
        return Model(
            inputs=[face_sequence_input, eye_sequence_input, mouth_sequence_input, numeric_input],
            outputs=output,
            name='temporal_model'
        )
    
    def _build_model(self) -> Model:
        """
        Xây dựng mô hình đầy đủ
        
        Returns:
            model: Mô hình hoàn chỉnh
        """
        # Xây dựng mô hình chính xử lý dữ liệu theo thời gian
        temporal_model = self._build_temporal_model()
        
        # Định nghĩa đầu vào
        face_sequence_input = Input(shape=(self.sequence_length, *self.face_shape, 3), name='face_sequence')
        eye_sequence_input = Input(shape=(self.sequence_length, *self.eye_shape, 3), name='eye_sequence')
        mouth_sequence_input = Input(shape=(self.sequence_length, *self.mouth_shape, 3), name='mouth_sequence')
        numeric_input = Input(shape=(self.sequence_length, 5), name='numeric_features')
        
        # Kết nối các đầu vào với mô hình xử lý thời gian
        output = temporal_model([face_sequence_input, eye_sequence_input, mouth_sequence_input, numeric_input])
        
        # Định nghĩa mô hình
        model = Model(
            inputs=[face_sequence_input, eye_sequence_input, mouth_sequence_input, numeric_input],
            outputs=output,
            name='drowsiness_detector'
        )
        
        # Biên dịch mô hình
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def train(self, 
              train_data: Dict[str, np.ndarray],
              validation_data: Dict[str, np.ndarray] = None,
              epochs: int = 50,
              batch_size: int = 32,
              callbacks: List = None) -> tf.keras.callbacks.History:
        """
        Huấn luyện mô hình
        
        Args:
            train_data: Dữ liệu huấn luyện dạng từ điển với các khóa:
                - 'face_sequence': Chuỗi ảnh khuôn mặt
                - 'eye_sequence': Chuỗi ảnh mắt
                - 'mouth_sequence': Chuỗi ảnh miệng
                - 'numeric_features': Chuỗi đặc trưng số
                - 'labels': Nhãn (1: buồn ngủ, 0: tỉnh táo)
            validation_data: Dữ liệu kiểm định (cùng định dạng với train_data)
            epochs: Số lượng epoch huấn luyện
            batch_size: Kích thước batch
            callbacks: Danh sách các callback
            
        Returns:
            history: Lịch sử huấn luyện
        """
        # Chuẩn bị dữ liệu đầu vào
        train_inputs = [
            train_data['face_sequence'],
            train_data['eye_sequence'],
            train_data['mouth_sequence'],
            train_data['numeric_features']
        ]
        
        train_labels = train_data['labels']
        
        val_inputs = None
        val_labels = None
        
        if validation_data is not None:
            val_inputs = [
                validation_data['face_sequence'],
                validation_data['eye_sequence'],
                validation_data['mouth_sequence'],
                validation_data['numeric_features']
            ]
            val_labels = validation_data['labels']
        
        # Huấn luyện mô hình
        history = self.model.fit(
            train_inputs,
            train_labels,
            validation_data=(val_inputs, val_labels) if validation_data is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return history
    
    def predict(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Dự đoán trạng thái buồn ngủ
        
        Args:
            data: Dữ liệu đầu vào dạng từ điển với các khóa:
                - 'face_sequence': Chuỗi ảnh khuôn mặt
                - 'eye_sequence': Chuỗi ảnh mắt
                - 'mouth_sequence': Chuỗi ảnh miệng
                - 'numeric_features': Chuỗi đặc trưng số
                
        Returns:
            predictions: Mảng các dự đoán (0-1)
        """
        # Chuẩn bị dữ liệu đầu vào
        inputs = [
            data['face_sequence'],
            data['eye_sequence'],
            data['mouth_sequence'],
            data['numeric_features']
        ]
        
        # Dự đoán
        predictions = self.model.predict(inputs)
        
        return predictions
    
    def save(self, filepath: str) -> None:
        """
        Lưu mô hình
        
        Args:
            filepath: Đường dẫn đến tệp lưu mô hình
        """
        self.model.save(filepath)
    
    def load(self, filepath: str) -> None:
        """
        Tải mô hình
        
        Args:
            filepath: Đường dẫn đến tệp mô hình
        """
        self.model = tf.keras.models.load_model(filepath)
        
    def evaluate(self, 
                 test_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Đánh giá mô hình
        
        Args:
            test_data: Dữ liệu kiểm tra dạng từ điển tương tự dữ liệu huấn luyện
            
        Returns:
            metrics: Từ điển chứa các metric đánh giá
        """
        # Chuẩn bị dữ liệu đầu vào
        test_inputs = [
            test_data['face_sequence'],
            test_data['eye_sequence'],
            test_data['mouth_sequence'],
            test_data['numeric_features']
        ]
        
        test_labels = test_data['labels']
        
        # Đánh giá mô hình
        results = self.model.evaluate(test_inputs, test_labels)
        
        # Tạo từ điển kết quả
        metrics_dict = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics_dict[metric_name] = results[i]
        
        return metrics_dict
        
    def summary(self) -> None:
        """
        Hiển thị tóm tắt mô hình
        """
        self.model.summary()

# Ví dụ code huấn luyện
from driver_monitoring.models.deep_learning import (
    FaceDataPreprocessor, 
    DrowsinessDetectorModel,
    DeepLearningModelManager
)
import numpy as np

# Tạo dữ liệu huấn luyện (giả sử đã tiền xử lý)
train_data = {
    'face_sequence': np.array([...]),  # shape: (samples, sequence_length, 128, 128, 3)
    'eye_sequence': np.array([...]),   # shape: (samples, sequence_length, 32, 128, 3)
    'mouth_sequence': np.array([...]), # shape: (samples, sequence_length, 32, 64, 3)
    'numeric_features': np.array([...]), # shape: (samples, sequence_length, 5)
    'labels': np.array([...])  # shape: (samples,)
}

# Khởi tạo model manager
model_manager = DeepLearningModelManager(models_dir='models/saved')

# Huấn luyện mô hình buồn ngủ
history = model_manager.train_drowsiness_model(
    train_data=train_data,
    validation_data=validation_data,
    epochs=50,
    batch_size=32,
    save_after_training=True
)

# Tương tự như mô hình buồn ngủ nhưng với dữ liệu khác nhau
train_data = {
    'face_sequence': np.array([...]),  # shape: (samples, sequence_length, 128, 128, 3)
    'eye_sequence': np.array([...]),   # shape: (samples, sequence_length, 32, 128, 3)
    'numeric_features': np.array([...]), # shape: (samples, sequence_length, 5)
    'labels': np.array([...])  # shape: (samples, 6) one-hot encoding
}

# Huấn luyện mô hình mất tập trung
history = model_manager.train_distraction_model(
    train_data=train_data,
    validation_data=validation_data,
    epochs=50,
    batch_size=32,
    save_after_training=True
) 