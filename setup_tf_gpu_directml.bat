@echo off

echo ============================
echo 📦 Установка TensorFlow и DirectML без автозависимостей
echo ============================
pip install tensorflow==2.10.0 --prefer-binary --no-deps
pip install tensorflow-directml-plugin==0.2.0.dev221020 --no-deps

echo ============================
echo 📦 Установка строго совместимых зависимостей
echo ============================
pip install keras==2.10.0 --no-deps
pip install tensorboard==2.10.0 --no-deps
pip install protobuf==3.19.6 --no-deps
pip install numpy==1.23.5 --no-deps
pip install wrapt==1.14.1 --no-deps
pip install absl-py==1.3.0 --no-deps
pip install gast==0.4.0 --no-deps
pip install six==1.16.0 --no-deps
pip install wheel==0.38.4 --no-deps
pip install opt_einsum==3.3.0 --no-deps
pip install astunparse==1.6.3 --no-deps
pip install termcolor==2.2.0 --no-deps
pip install flatbuffers==1.12 --no-deps
pip install requests==2.28.2 --no-deps
pip install urllib3==1.26.15 --no-deps
pip install charset_normalizer==3.1.0 --no-deps
pip install idna==3.4 --no-deps
pip install certifi==2023.5.7 --no-deps
pip install --upgrade tensorflow-directml-plugin
pip install pandas 
pip install scikit-learn
pip install matplotlib
pip install mysql-connector-python
pip install dotenv
pip install ta
pip install tzlocal
 pip install plotly

REM pip install h5py==3.7.0 --no-deps
REM pip install google-pasta==0.2.0 --no-deps
REM pip install typing-extensions==4.3.0 --no-deps


echo ============================
echo 🧪 Создание файла test_gpu.py
echo ============================
echo import tensorflow as tf > test_gpu.py
echo print("TensorFlow version:", tf.__version__) >> test_gpu.py
echo gpus = tf.config.list_physical_devices('GPU') >> test_gpu.py
echo if not gpus: >> test_gpu.py
echo     print("🚫 GPU не обнаружена. TensorFlow работает на CPU.") >> test_gpu.py
echo else: >> test_gpu.py
echo     print(f"✅ Обнаружено {len(gpus)} GPU:") >> test_gpu.py
echo     for i, gpu in enumerate(gpus): >> test_gpu.py
echo         print(f"  GPU {i}: {gpu}") >> test_gpu.py
echo     with tf.device('/GPU:0'): >> test_gpu.py
echo         a = tf.random.normal([1000, 1000]) >> test_gpu.py
echo         b = tf.random.normal([1000, 1000]) >> test_gpu.py
echo         c = tf.matmul(a, b) >> test_gpu.py
echo     print("🚀 Успешно выполнено матричное умножение на GPU.") >> test_gpu.py

echo ============================
echo 🚀 Запуск теста GPU
echo ============================
python test_gpu.py

pause
