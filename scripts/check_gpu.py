#!/usr/bin/env python3
"""
Script kiểm tra GPU availability và cấu hình TensorFlow
"""

import sys
import os

def check_tensorflow():
    """Kiểm tra TensorFlow installation"""
    print("🔍 Checking TensorFlow installation...")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        return tf
    except ImportError:
        print("❌ TensorFlow not installed")
        print("Install with: pip install tensorflow[and-cuda]")
        return None

def check_cuda_cudnn(tf):
    """Kiểm tra CUDA và cuDNN"""
    print("\n🔍 Checking CUDA/cuDNN...")
    cuda_built = tf.test.is_built_with_cuda()
    cudnn_built = tf.test.is_built_with_cudnn()

    print(f"CUDA built: {'✅' if cuda_built else '❌'}")
    print(f"cuDNN built: {'✅' if cudnn_built else '❌'}")

    if not cuda_built:
        print("⚠️  TensorFlow was not built with CUDA support")
        print("Install GPU version: pip install tensorflow[and-cuda]")

    return cuda_built and cudnn_built

def check_gpu_devices(tf):
    """Kiểm tra GPU devices"""
    print("\n🔍 Checking GPU devices...")

    # List physical devices
    gpus = tf.config.experimental.list_physical_devices('GPU')
    cpus = tf.config.experimental.list_physical_devices('CPU')

    print(f"CPU devices: {len(cpus)}")
    for cpu in cpus:
        print(f"  - {cpu}")

    print(f"GPU devices: {len(gpus)}")
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  - GPU {i}: {gpu}")
            try:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"    └─ {details}")
            except:
                pass
    else:
        print("❌ No GPU devices found")
        return False

    # Test GPU availability
    gpu_available = tf.test.is_gpu_available()
    print(f"GPU available for TensorFlow: {'✅' if gpu_available else '❌'}")

    return len(gpus) > 0 and gpu_available

def test_gpu_computation(tf):
    """Test GPU computation"""
    print("\n🔍 Testing GPU computation...")

    if not tf.test.is_gpu_available():
        print("❌ GPU not available for computation")
        return False

    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 0.0], [0.0, 1.0]])
            c = tf.matmul(a, b)

            result = c.numpy()
            print("✅ GPU computation successful")
            print(f"Result: {result}")
            return True

    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        return False

def show_recommendations():
    """Hiển thị recommendations"""
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS")
    print("="*60)

    print("1. GPU Training Commands:")
    print("   python train_level1_cnn.py --mixed-precision --xla --gpu-memory-limit 8")

    print("\n2. Multi-GPU Training:")
    print("   python train_level1_cnn.py --gpu-device '0,1' --batch-size 128")

    print("\n3. Memory Limited Systems:")
    print("   python train_level1_cnn.py --gpu-memory-limit 4 --batch-size 32")

    print("\n4. Debug GPU Issues:")
    print("   python scripts/check_gpu.py")

def main():
    """Main function"""
    print("🚀 GPU Configuration Check for Safenet IDS")
    print("="*50)

    # Check TensorFlow
    tf = check_tensorflow()
    if not tf:
        sys.exit(1)

    # Check CUDA/cuDNN
    cuda_ok = check_cuda_cudnn(tf)

    # Check GPU devices
    gpu_ok = check_gpu_devices(tf)

    # Test computation
    if gpu_ok:
        test_gpu_computation(tf)

    # Show status summary
    print("\n" + "="*50)
    print("📊 STATUS SUMMARY")
    print("="*50)

    status = []
    status.append(("TensorFlow", "✅ Installed" if tf else "❌ Missing"))
    status.append(("CUDA", "✅ Built-in" if cuda_ok else "❌ Not built"))
    status.append(("GPU Devices", "✅ Found" if gpu_ok else "❌ Not found"))
    status.append(("GPU Computation", "✅ Working" if gpu_ok and test_gpu_computation(tf) else "❌ Failed"))

    for item, status_str in status:
        print(f"{item:15}: {status_str}")

    # Overall recommendation
    if gpu_ok and cuda_ok:
        print("\n🎉 GPU is ready for training!")
        print("Use --mixed-precision --xla for best performance")
    else:
        print("\n⚠️  GPU training not available")
        print("Training will be slow on CPU")

    show_recommendations()

if __name__ == "__main__":
    main()
