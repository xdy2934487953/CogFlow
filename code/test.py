# 与项目无关环境测试
import tensorflow as tf
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 强制使用第一块GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # 防止显存预分配错误
print("Python版本:", sys.version)
print("TensorFlow版本:", tf.__version__)
print("GPU可用:", tf.config.list_physical_devices('GPU'))


# 测试一个简单的GPU操作
with tf.device('/GPU:0'):
    a = tf.constant([1.0, 2.0, 3.0])
    b = tf.constant([4.0, 5.0, 6.0])
    c = a + b
    print("GPU计算测试结果:", c.numpy())