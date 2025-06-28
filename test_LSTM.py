# import tensorflow as tf
# import numpy as np

# # Включаем логирование размещения операций
# tf.debugging.set_log_device_placement(True)

# # Параметры
# window_size = 240
# feature_count = 17
# batch_size = 64
# neyro = 64
# epochs = 30

# # Генерация случайных данных (float32)
# X_train = np.random.rand(1000, window_size, feature_count).astype(np.float32)
# y_train = np.random.rand(1000, 1).astype(np.float32)
# X_val = np.random.rand(200, window_size, feature_count).astype(np.float32)
# y_val = np.random.rand(200, 1).astype(np.float32)

# # Создание модели с implementation=1 для cuDNN LSTM
# model = tf.keras.Sequential([
#     tf.keras.layers.LSTM(neyro, return_sequences=True, input_shape=(window_size, feature_count), implementation=1),
#     tf.keras.layers.LSTM(neyro // 2, implementation=1),
#     tf.keras.layers.Dense(1)
# ])

# model.compile(optimizer='adam', loss='mse')

# # Обучение модели напрямую на numpy-массивах
# history = model.fit(
#     X_train,
#     y_train,
#     batch_size=batch_size,
#     epochs=epochs,
#     validation_data=(X_val, y_val)
# )
# test_gpu.py
import tensorflow as tf
import time

print("GPU доступен:", tf.config.list_physical_devices('GPU'))

start = time.time()
# Простая модель для проверки
model = tf.keras.Sequential([tf.keras.layers.Dense(64, input_shape=(10,))])
model.compile(loss='mse')
model.fit(tf.random.normal((1000, 10)), tf.random.normal((1000, 1)), epochs=5)
print(f"Время на эпоху: {(time.time() - start)/5:.2f} сек")