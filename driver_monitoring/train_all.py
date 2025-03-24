import os
import sys
import argparse
import subprocess
import time

def print_header(text):
    """In tiêu đề được trang trí"""
    print("\n" + "="*80)
    print(f"    {text}")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện toàn bộ hệ thống giám sát tài xế")
    parser.add_argument("--drowsiness_data", type=str, default="datasets/drowsiness",
                        help="Đường dẫn đến thư mục dữ liệu phát hiện buồn ngủ")
    parser.add_argument("--distraction_data", type=str, default="datasets/distraction",
                        help="Đường dẫn đến thư mục dữ liệu phát hiện mất tập trung")
    parser.add_argument("--models_dir", type=str, default="models/saved",
                        help="Đường dẫn đến thư mục lưu mô hình")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Số lượng epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Kích thước batch")
    parser.add_argument("--only_drowsiness", action="store_true",
                        help="Chỉ huấn luyện mô hình phát hiện buồn ngủ")
    parser.add_argument("--only_distraction", action="store_true",
                        help="Chỉ huấn luyện mô hình phát hiện mất tập trung")
    
    args = parser.parse_args()
    
    # Tạo thư mục lưu mô hình nếu chưa tồn tại
    os.makedirs(args.models_dir, exist_ok=True)
    
    print_header("HƯỚNG DẪN HUẤN LUYỆN MÔ HÌNH GIÁM SÁT TÀI XẾ")
    
    print("Hướng dẫn này sẽ giúp bạn huấn luyện các mô hình deep learning cho hệ thống giám sát tài xế.")
    print("Có hai mô hình chính cần huấn luyện:")
    print("1. Mô hình phát hiện buồn ngủ (drowsiness_detector)")
    print("2. Mô hình phát hiện mất tập trung (distraction_detector)")
    print("\nLưu ý: Quá trình huấn luyện có thể mất nhiều thời gian tùy thuộc vào cấu hình máy tính và kích thước dữ liệu.")
    
    # Hiển thị cấu trúc thư mục dữ liệu mong đợi
    print_header("CẤU TRÚC THƯ MỤC DỮ LIỆU")
    
    print("1. Dữ liệu phát hiện buồn ngủ:")
    print(f"   {args.drowsiness_data}/")
    print("   ├── alert/   - Chứa video/hình ảnh tài xế tỉnh táo")
    print("   └── drowsy/  - Chứa video/hình ảnh tài xế buồn ngủ")
    
    print("\n2. Dữ liệu phát hiện mất tập trung:")
    print(f"   {args.distraction_data}/")
    print("   ├── focused/       - Tài xế tập trung")
    print("   ├── talking/       - Tài xế nói chuyện")
    print("   ├── phone_use/     - Tài xế dùng điện thoại")
    print("   ├── texting/       - Tài xế nhắn tin")
    print("   ├── radio_adjust/  - Tài xế điều chỉnh radio/thiết bị")
    print("   └── looking_away/  - Tài xế nhìn ra ngoài")
    
    # Kiểm tra thư mục dữ liệu
    print_header("KIỂM TRA DỮ LIỆU")
    
    # Kiểm tra dữ liệu phát hiện buồn ngủ
    drowsiness_data_valid = False
    if not args.only_distraction:
        if os.path.exists(args.drowsiness_data):
            alert_dir = os.path.join(args.drowsiness_data, "alert")
            drowsy_dir = os.path.join(args.drowsiness_data, "drowsy")
            
            if os.path.exists(alert_dir) and os.path.exists(drowsy_dir):
                print(f"✓ Thư mục dữ liệu phát hiện buồn ngủ hợp lệ: {args.drowsiness_data}")
                drowsiness_data_valid = True
            else:
                print(f"✗ Thư mục dữ liệu phát hiện buồn ngủ không đúng cấu trúc: {args.drowsiness_data}")
                print(f"  Không tìm thấy thư mục 'alert' hoặc 'drowsy'")
        else:
            print(f"✗ Không tìm thấy thư mục dữ liệu phát hiện buồn ngủ: {args.drowsiness_data}")
    
    # Kiểm tra dữ liệu phát hiện mất tập trung
    distraction_data_valid = False
    if not args.only_drowsiness:
        if os.path.exists(args.distraction_data):
            required_dirs = ["focused", "talking", "phone_use", "texting", "radio_adjust", "looking_away"]
            missing_dirs = [d for d in required_dirs if not os.path.exists(os.path.join(args.distraction_data, d))]
            
            if not missing_dirs:
                print(f"✓ Thư mục dữ liệu phát hiện mất tập trung hợp lệ: {args.distraction_data}")
                distraction_data_valid = True
            else:
                print(f"✗ Thư mục dữ liệu phát hiện mất tập trung không đúng cấu trúc: {args.distraction_data}")
                print(f"  Không tìm thấy các thư mục: {', '.join(missing_dirs)}")
        else:
            print(f"✗ Không tìm thấy thư mục dữ liệu phát hiện mất tập trung: {args.distraction_data}")
    
    # Huấn luyện các mô hình
    if (args.only_drowsiness and not drowsiness_data_valid) or \
       (args.only_distraction and not distraction_data_valid) or \
       (not args.only_drowsiness and not args.only_distraction and not (drowsiness_data_valid or distraction_data_valid)):
        print_header("LỖI")
        print("Không thể tiếp tục vì không tìm thấy dữ liệu hợp lệ.")
        print("Vui lòng chuẩn bị dữ liệu theo cấu trúc yêu cầu và thử lại.")
        return
    
    # Huấn luyện mô hình phát hiện buồn ngủ
    if drowsiness_data_valid and (not args.only_distraction):
        print_header("HUẤN LUYỆN MÔ HÌNH PHÁT HIỆN BUỒN NGỦ")
        
        print(f"Bắt đầu huấn luyện mô hình phát hiện buồn ngủ với {args.epochs} epochs, batch size {args.batch_size}")
        cmd = [
            "python", "train_drowsiness.py",
            "--data_dir", args.drowsiness_data,
            "--models_dir", args.models_dir,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size)
        ]
        
        print(f"Lệnh thực thi: {' '.join(cmd)}")
        subprocess.run(cmd)
    
    # Huấn luyện mô hình phát hiện mất tập trung
    if distraction_data_valid and (not args.only_drowsiness):
        print_header("HUẤN LUYỆN MÔ HÌNH PHÁT HIỆN MẤT TẬP TRUNG")
        
        print(f"Bắt đầu huấn luyện mô hình phát hiện mất tập trung với {args.epochs} epochs, batch size {args.batch_size}")
        cmd = [
            "python", "train_distraction.py",
            "--data_dir", args.distraction_data,
            "--models_dir", args.models_dir,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size)
        ]
        
        print(f"Lệnh thực thi: {' '.join(cmd)}")
        subprocess.run(cmd)
    
    print_header("HƯỚNG DẪN CHẠY HỆ THỐNG")
    
    print("Sau khi huấn luyện xong, bạn có thể chạy hệ thống bằng các cách sau:")
    
    print("\n1. Chạy ứng dụng trực tiếp sử dụng camera:")
    print("   python -m app.main --camera 0")
    print("   Các tùy chọn:")
    print("   --camera ID     : ID của camera (mặc định: 0)")
    print("   --no-ui         : Không hiển thị giao diện người dùng")
    print("   --record        : Ghi lại video")
    print("   --output DIR    : Thư mục đầu ra (mặc định: output)")
    
    print("\n2. Chạy API để tích hợp với ứng dụng khác:")
    print("   uvicorn app.api:app --host 0.0.0.0 --port 8000")
    print("   API sẽ chạy tại địa chỉ http://localhost:8000")
    print("   Tài liệu API tự động: http://localhost:8000/docs")
    
    print("\n3. Sử dụng mô hình trong code của bạn:")
    print("   from models.deep_learning import DeepLearningModelManager")
    print("   model_manager = DeepLearningModelManager(models_dir='models/saved', load_models=True)")
    print("   # Xử lý frame")
    print("   results = model_manager.process_frame(frame, face_landmarks, left_ear, right_ear, mar, roll, pitch, yaw)")
    
    print_header("HOÀN THÀNH")
    print("Cảm ơn bạn đã sử dụng hệ thống giám sát tài xế bằng AI!")

if __name__ == "__main__":
    main() 