#!/usr/bin/env python3
"""
Demo script để show cách tối ưu hóa GPU training cho CNN+LSTM models
"""

import subprocess
import sys
from pathlib import Path

def show_gpu_options():
    """Hiển thị các tùy chọn GPU optimization"""
    print("=" * 80)
    print("🚀 HƯỚNG DẪN TỐI ƯU HÓA GPU TRAINING CHO CNN+LSTM")
    print("=" * 80)

    print("\n📋 CÁC TÙY CHỌN TĂNG TỐC ĐỘ TRAINING:")
    print("1. --mixed-precision: Sử dụng float16 thay vì float32")
    print("   - Tăng tốc 2-3x trên GPU")
    print("   - Giảm memory usage 50%")
    print("   - Độ chính xác tương đương")

    print("\n2. --xla: Enable XLA (Accelerated Linear Algebra)")
    print("   - Tối ưu hóa graph execution")
    print("   - Tăng tốc 10-50% trên GPU")

    print("\n3. --gpu-memory-limit X: Giới hạn GPU memory")
    print("   - Ví dụ: --gpu-memory-limit 8 (8GB)")
    print("   - Tránh out-of-memory errors")

    print("\n4. --gpu-device '0,1': Chỉ định GPU cụ thể")
    print("   - Multi-GPU training")
    print("   - Load balancing")

    print("\n💡 KẾT HỢP TỐI ƯU:")
    print("   --mixed-precision --xla --gpu-memory-limit 8")

def run_training_demo():
    """Demo training với GPU optimization"""

    print("\n" + "="*60)
    print("🎯 DEMO TRAINING VỚI GPU OPTIMIZATION")
    print("="*60)

    # Kiểm tra xem có file dataset không
    dataset_file = "dataset_clean_cnn.pkl"
    if not Path(dataset_file).exists():
        print(f"❌ Không tìm thấy {dataset_file}")
        print("Chạy scripts/preprocess_dataset.py --model-type both trước")
        return

    # Ví dụ training với GPU optimization
    cmd = [
        sys.executable, "ids_pipeline/_1d_cnn/train_level1_cnn.py",
        "--source-dataset", dataset_file,
        "--output-dir", "artifacts_cnn_gpu_demo",
        "--epochs", "5",  # Ít epochs cho demo
        "--batch-size", "64",
        "--mixed-precision",
        "--xla",
        "--gpu-memory-limit", "4",  # Giới hạn 4GB cho demo
        "--auto-split"
    ]

    print("Chạy lệnh:")
    print(" ".join(cmd))
    print("\n📊 Mong đợi:")
    print("- Training trên GPU với mixed precision")
    print("- XLA optimization enabled")
    print("- Memory limit 4GB")
    print("- Early stopping nếu có GPU")

    try:
        print("\n🚀 Đang chạy training demo...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ Training demo hoàn thành!")
        else:
            print(f"⚠️  Training kết thúc với code: {result.returncode}")
            print("Log:", result.stderr[-500:])  # Last 500 chars

    except subprocess.TimeoutExpired:
        print("⏰ Training demo timeout (5 phút)")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

def show_system_info():
    """Hiển thị thông tin system và GPU"""
    print("\n" + "="*60)
    print("💻 THÔNG TIN SYSTEM & GPU")
    print("="*60)

    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        print(f"CUDA built: {tf.test.is_built_with_cuda()}")
        print(f"cuDNN built: {tf.test.is_built_with_cudnn()}")

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU found: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                details = tf.config.experimental.get_device_details(gpu)
                print(f"  GPU {i}: {details.get('device_name', 'Unknown')}")
        else:
            print("❌ No GPU found - training will be slow on CPU")

    except ImportError:
        print("❌ TensorFlow not installed")

def main():
    show_gpu_options()
    show_system_info()
    run_training_demo()

if __name__ == "__main__":
    main()
