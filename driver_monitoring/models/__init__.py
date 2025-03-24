from .face_detector import FaceDetector
from .drowsiness_detector import DrowsinessDetector
from .distraction_detector import DistractionDetector
from .dangerous_behavior_detector import DangerousBehaviorDetector

# Thử import deep learning modules, nhưng không gây lỗi nếu thiếu thư viện
try:
    from .deep_learning import (
        FaceDataPreprocessor,
        DrowsinessDetectorModel,
        DistractionDetectorModel,
        DeepLearningModelManager
    )
    has_deep_learning = True
except ImportError:
    has_deep_learning = False

__all__ = [
    'FaceDetector',
    'DrowsinessDetector',
    'DistractionDetector',
    'DangerousBehaviorDetector'
]

if has_deep_learning:
    __all__.extend([
        'FaceDataPreprocessor',
        'DrowsinessDetectorModel',
        'DistractionDetectorModel',
        'DeepLearningModelManager'
    ]) 