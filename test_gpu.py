import tensorflow as tf 
print("TensorFlow version:", tf.__version__) 
gpus = tf.config.list_physical_devices('GPU') 
if not gpus: 
    print("🚫 GPU не обнаружена. TensorFlow работает на CPU.") 
else: 
    print(f"✅ Обнаружено {len(gpus)} GPU:") 
    for i, gpu in enumerate(gpus): 
        print(f"  GPU {i}: {gpu}") 
    with tf.device('/GPU:0'): 
        a = tf.random.normal([1000, 1000]) 
        b = tf.random.normal([1000, 1000]) 
        c = tf.matmul(a, b) 
    print("🚀 Успешно выполнено матричное умножение на GPU.") 
