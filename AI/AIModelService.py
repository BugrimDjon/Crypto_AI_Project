# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # номер NVIDIA GPU
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Conv1D, MaxPooling1D, Bidirectional, BatchNormalization, SpatialDropout1D
from tensorflow.keras import Input, Sequential, regularizers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
from tensorflow.keras.models import load_model, Sequential
import tensorflow as tf

from enums.coins import Coins
from database.db import Database
from services.okx_candles import OkxCandlesFetcher
from services.time_control import TimControl
from enums.timeframes import Timeframe
from config.SettingsCoins import SettingsCoins
from enums.AfterBefore import AfterBefore
from logger.context_logger import ContextLogger
import pickle
from pympler import muppy, summary

import pandas as pd
import ta
import json
import logging

import tensorflow as tf
from tensorflow.keras import mixed_precision
from sklearn.preprocessing import StandardScaler
import logging
import gc
from tensorflow.keras.callbacks import Callback
import GPUtil
import psutil
import time


# tf.debugging.experimental.enable_dump_debug_info(
#     "/tmp/tf_logs",
#     tensor_debug_mode="FULL_HEALTH",
#     circular_buffer_size=-1
# )



# Настройка логирования
logging.basicConfig(
    level=logging.INFO,  # или DEBUG, WARNING, ERROR
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),  # Писать в файл
        logging.StreamHandler(),  # Писать в консоль
    ],
)

class SystemMonitorCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print()
        # Получаем температуру и загрузку GPU
        gpus = GPUtil.getGPUs()
        gpu_info = []
        for gpu in gpus:
            gpu_info.append(
                f"GPU {gpu.id} ({gpu.name}): Temp={gpu.temperature}°C, Load={gpu.load*100:.1f}%, Mem={gpu.memoryUtil*100:.1f}%"
            )
        gpu_status = " | ".join(gpu_info) if gpu_info else "GPU info not available"

        # Получаем температуру CPU (если доступна)
        temps = psutil.sensors_temperatures()
        cpu_temps = temps.get('coretemp') or temps.get('cpu_thermal') or []
        cpu_temp_str = ", ".join([f"{t.label or 'CPU'}: {t.current}°C" for t in cpu_temps]) if cpu_temps else "CPU temp info not available"

        print(f"[Epoch {epoch + 1}] {gpu_status} | {cpu_temp_str}")
        if gpu.temperature>70:
            time.sleep(20)



class AIModelService:
    def __init__(self, db: Database) -> None:
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.db = db

    def train_model_experiment(
    self,
    table_name: Coins,
    time_frame: Timeframe,
    limit: int = 1000000,
    window_size: int = 60,
    horizon: int = 1,
    l2_reg=None,
    model_path=None,
    scaler_path=None,
    return_predictions=False,
    epochs: int = 50,
    learning_rate: float = 0.001,
    dropout: float = 0.2,
    neyro: int = 64,
    df_ready=None,
    offset=None,
    batch_size=64,
    target_type =0,
):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[0], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError as e:
                print("Ошибка при настройке GPU:", e)

        mixed_precision.set_global_policy('mixed_float16')

        # === 1. Загрузка данных ===
        if df_ready is None:
            query = f"""SELECT ts, o, h, l, c,
                            vol, volCcy, volCcyQuote,
                            ma50, ma200, ema12, ema26,
                            macd, macd_signal, rsi14, macd_histogram,
                            stochastic_k, stochastic_d
                        FROM {table_name.value}
                        WHERE timeFrame=%s AND ts >= %s
                        ORDER BY ts ASC
                        LIMIT %s;"""
            params = (time_frame.label, 0, limit)
            rows = self.db.query_to_bd(query, params)

            columns = [
                "ts", "o", "h", "l", "c", "vol", "volCcy", "volCcyQuote", "ma50", "ma200", "ema12",
                "ema26", "macd", "macd_signal", "rsi14", "macd_histogram", "stochastic_k", "stochastic_d"
            ]
            df = pd.DataFrame(rows, columns=columns)
        else:
            df = df_ready.copy()
            df = df.drop(columns=["offset"], errors="ignore")
            

        # === 2. Формирование таргета ===
        if target_type == 0:
            df["target"] = np.log(df["c"].shift(-horizon) / df["c"])
            inverse_transform = lambda pred, now: now * np.exp(pred)
        elif target_type == 1:
            df["target"] = np.log(df["c"].shift(-horizon))
            inverse_transform = lambda pred, now: np.exp(pred)
        elif target_type == 2:
            df["target"] = (df["c"].shift(-horizon) - df["c"]) / df["c"]
            inverse_transform = lambda pred, now: now * (1 + pred)
        elif target_type == 3:
            df["target"] = df["c"].shift(-horizon) - df["c"]
            inverse_transform = lambda pred, now: now + pred
        elif target_type == 4:
            df["target"] = df["c"].shift(-horizon)
            inverse_transform = lambda pred, now: pred

        df.dropna(inplace=True)

       
       
        # Используем относительное изменение цены (в процентах): (future - now) / now
        # df["target"] = (df["c"].shift(-horizon) - df["c"]) / df["c"]
        # df["target"] = np.log(df["c"].shift(-horizon) / df["c"])
        # df["target"] = np.log(df["c"].shift(-horizon)) 

        
        # Подменим c
        replace_c = False
        if replace_c:
            df["c"] = 100
        # Конец  Подменим c

        features = df.drop(columns=["ts", "target", "c"])
        # === 3. Масштабирование ===

        # Масштабируем
        feature_scaler = MinMaxScaler()
        features_scaled = feature_scaler.fit_transform(features)
        if target_type in [1, 4]:
            target_scaler = MinMaxScaler()
            y_scaled = target_scaler.fit_transform(df["target"].values.reshape(-1, 1))
        else:
            target_scaler = None
            y_scaled = df["target"].values.reshape(-1, 1)


        # === 4. Формирование последовательностей ===
        def create_sequences(X, y, window_size, horizon):
            Xs, ys = [], []
            for i in range(len(X) - window_size - horizon + 1):
                Xs.append(X[i : (i + window_size)])
                ys.append(y[i + window_size + horizon - 1])
            return np.array(Xs), np.array(ys)

        X_lstm, y_lstm = create_sequences(features_scaled, df["target"].values.reshape(-1, 1), window_size, horizon)

        # === 5. Train/Test Split ===
        split_idx = int(0.8 * len(X_lstm))
        X_train, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
        y_train, y_test = y_lstm[:split_idx], y_lstm[split_idx:]

        # Приводим к нужному типу
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_test = y_test.astype(np.float32)

        # Перемешиваем обучающую выборку
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train, y_train = X_train[indices], y_train[indices]

        # === 6. Dataset и batching ===
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

        # === 7. Создание модели ===
        # model = Sequential([
        #     Input(shape=(X_train.shape[1], X_train.shape[2])),
        #     Conv1D(filters=32, kernel_size=3, activation="relu"),
        #     MaxPooling1D(pool_size=2),
        #     Bidirectional(GRU(neyro, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg), implementation=1)),
        #     Bidirectional(GRU(neyro // 2, return_sequences=False, kernel_regularizer=regularizers.l2(l2_reg), implementation=1)),
        #     Dropout(dropout),
        #     Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)),
        #     Dense(1, dtype='float32')  # Прогноз относительного изменения
        # ])

        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            Conv1D(filters=16, kernel_size=3, activation="relu"),
            BatchNormalization(),
            SpatialDropout1D(dropout),
            MaxPooling1D(pool_size=2),

            Bidirectional(GRU(neyro, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg))),
            SpatialDropout1D(dropout),

            Bidirectional(GRU(neyro // 2, return_sequences=False, kernel_regularizer=regularizers.l2(l2_reg))),
            Dropout(dropout),  # 🔹 Оставляем и здесь тоже (финальный слой)

            Dense(16, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)),
            Dense(1, dtype='float32')
        ])


        model.compile(optimizer=Adam(learning_rate), loss="mse")

        # === 8. Callbacks ===
        early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

        # === 9. Обучение ===
        try:
            history = model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset,
                callbacks=[early_stop, SystemMonitorCallback(), reduce_lr],
                verbose=1,
            )
        except Exception as e:
            print(f"Ошибка при обучении: {e}")
            tf.keras.backend.clear_session()
            gc.collect()
            raise

        # === 10. Оценка и сохранение ===
        loss = model.evaluate(X_test, y_test)
        print(f"Final loss (MSE) on test set: {loss}")

        model.save(model_path)
        with open(scaler_path, "wb") as f:
            pickle.dump((feature_scaler, target_scaler), f)  # Сохраняем только feature_scaler

        # === 11. Прогнозы (если нужно вернуть) ===
        if return_predictions:
            y_pred = model.predict(X_test, batch_size=batch_size)

            if target_type in [1, 4] and target_scaler is not None:
                y_pred = target_scaler.inverse_transform(y_pred)
                y_test = target_scaler.inverse_transform(y_test)

            # Берем последние цены close на момент прогноза
            close_test_now = df["c"].values[split_idx + window_size - 1 : split_idx + window_size - 1 + len(y_test)]

            # Восстанавливаем реальные будущие цены и предсказанные
            if target_type == 0:
                # log(future / now) => future = now * exp(pred)
                predicted_price = close_test_now * np.exp(y_pred.flatten())
                true_price = close_test_now * np.exp(y_test.flatten())
            elif target_type == 1:
                # log(future) => future = exp(pred)
                predicted_price = np.exp(y_pred.flatten())
                true_price = np.exp(y_test.flatten())
            elif target_type == 2:
                # (future - now) / now => future = now * (1 + pred)
                predicted_price = close_test_now * (1 + y_pred.flatten())
                true_price = close_test_now * (1 + y_test.flatten())
            elif target_type == 3:
                # future - now => future = now + pred
                predicted_price = close_test_now + y_pred.flatten()
                true_price = close_test_now + y_test.flatten()
            elif target_type == 4:
                # target = future => предсказание — это уже цена
                predicted_price = y_pred.flatten()
                true_price = y_test.flatten()


            # Логируем корреляции
            corr_with_future = np.corrcoef(predicted_price, true_price)[0, 1]
            corr_with_now = np.corrcoef(predicted_price, close_test_now)[0, 1]

            print(f"📈 Корреляция прогноза с реальной будущей ценой: {corr_with_future:.4f}")
            print(f"📉 Корреляция прогноза с текущей (входной) ценой: {corr_with_now:.4f}")
            


            # y_pred — относительное изменение, преобразовывать не нужно
            del features, features_scaled, df
            gc.collect()
            return model, history, batch_size, feature_scaler, None, y_test, y_pred, corr_with_future, corr_with_now

        return model, history, batch_size, feature_scaler, None






    def train_model_experiment_big_korelaciya(
        self,
        table_name: Coins,
        time_frame: Timeframe,
        limit: int = 1000000,
        window_size: int = 60,
        horizon: int = 1,
        l2_reg=None,
        model_path=None,
        scaler_path=None,
        return_predictions=False,
        epochs: int = 50,
        learning_rate: float = 0.001,  # регулируем
        dropout: float = 0.2,  # регулируем
        neyro: int = 64,
        df_ready=None,
        offset=None,
        batch_size=64
    ):
        
        # tf.debugging.set_log_device_placement(True)  # Включаем логирование размещения операций

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[0], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError as e:
                print("Ошибка при настро    йке GPU:", e)
        
        mixed_precision.set_global_policy('mixed_float16')

        if df_ready is None:
            query = f""" SELECT ts, o, h, l, c,
                        vol, volCcy, volCcyQuote,
                        ma50, ma200, ema12, ema26,
                        macd, macd_signal, rsi14, macd_histogram,
                        stochastic_k, stochastic_d
                    FROM {table_name.value}
                            where timeFrame=%s and ts>=%s
                            ORDER BY ts ASC
                            limit %s;
                    """
            params = (time_frame.label, 0, limit)
            rows = self.db.query_to_bd(query, params)

            columns = [
                "ts","o", "h", "l", "c", "vol", "volCcy", "volCcyQuote", "ma50", "ma200", "ema12",
                "ema26", "macd", "macd_signal", "rsi14", "macd_histogram", "stochastic_k", "stochastic_d", ]
            df = pd.DataFrame(rows, columns=columns)
        else:
            df=df_ready.copy()
            df = df.drop(columns=["offset"])

        # Целевая — следующая цена закрытия
        # df["target"] = df["c"].shift(-horizon) # сильно коррелирует с текущей ценой
        #df["target"] = np.log(df["c"].shift(-horizon)) - np.log(df["c"])  # применяем  логарифмическую разницу цены закрытия
        df["target"] = (df["c"].shift(-horizon) - df["c"]) / df["c"]
        
        df.dropna(inplace=True)

        features = df.drop(columns=["ts", "target"])

        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        # feature_scaler = StandardScaler()
        # target_scaler = StandardScaler()
        # print("target_scaler = StandardScaler()")

        # масштабирование
        features_scaled = feature_scaler.fit_transform(features)
        target_scaled = target_scaler.fit_transform(df[["target"]])
        # print("target_scaled = target_scaler.fit_transform(df[['target']])")

        def create_sequences(X, y, window_size, horizon):
            Xs, ys = [], []
            for i in range(len(X) - window_size - horizon + 1):
                Xs.append(X[i : (i + window_size)])
                ys.append(y[i + window_size + horizon - 1])
            return np.array(Xs), np.array(ys)

        # window_size = 60
        X_lstm, y_lstm = create_sequences(
            features_scaled, target_scaled, window_size, horizon
        )
        # print("X_lstm, y_lstm = create_sequences")
        
        split_idx = int(0.8 * len(X_lstm))
        X_train, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
        y_train, y_test = y_lstm[:split_idx], y_lstm[split_idx:]

        del X_lstm, y_lstm


        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_test = y_test.astype(np.float32)

        # ---- Добавлено: перемешивание обучающей выборки ----
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
        # ------------------------------------------------------
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
      
        # # Начало Линивая загрузка
        # def data_generator():
        #     for x, y in zip(X_train.astype(np.float16), y_train.astype(np.float16)):
        #         yield x, y

        # train_dataset = tf.data.Dataset.from_generator(
        #     data_generator,
        #     output_signature=(
        #         tf.TensorSpec(shape=(X_train.shape[1], X_train.shape[2]), dtype=tf.float32),
        #         tf.TensorSpec(shape=(1,), dtype=tf.float32),
        #     )
        # )
        # # Конец Линивая загрузка

        # train_dataset = train_dataset.cache()
        # train_dataset = train_dataset.cache(filename='cache.tf-data')

        # перемешиваем Выбрали перемешивание выше
        # train_dataset = train_dataset.shuffle(buffer_size=1024)

        # формируем батчи
        train_dataset = train_dataset.batch(batch_size)

        # print("→ prefetch")
        # train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        # print("Делаем val_dataset = val_dataset.batch(batch_size)")
        val_dataset = val_dataset.batch(batch_size)
        # val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        # model = Sequential([
        #     Input(shape=(X_train.shape[1], X_train.shape[2])),
        #     LSTM(neyro, return_sequences=True, implementation=1),
        #     LSTM(neyro // 2, return_sequences=False, implementation=1),
        #     Dropout(dropout),
        #     Dense(32, activation="relu"),
        #     Dense(1, dtype='float32')
        # ])
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            GRU(neyro, return_sequences=True, 
                kernel_regularizer=regularizers.l2(l2_reg), implementation=1),
            GRU(neyro // 2, return_sequences=False, 
                kernel_regularizer=regularizers.l2(l2_reg), implementation=1),
            Dropout(dropout),
            Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)),
            Dense(1, dtype='float32')
        ])

        model.compile(optimizer=Adam(learning_rate), loss="mse")

        early_stop = EarlyStopping(
            monitor="val_loss", patience=30, restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )

        # print("=== Начинаем model.fit ===")
        try:
            history = model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset,
                callbacks=[early_stop, SystemMonitorCallback(), reduce_lr ],
                verbose=1,
            )
        except Exception as e:
            import traceback
            print(f"Ошибка при обучении: {e}")
            traceback.print_exc()
            # Освобождаем ресурсы при ошибке
            tf.keras.backend.clear_session()
            gc.collect()
            raise  # Перебрасываем исключение дальше
        # print("=== Обучение завершено ===")
        loss = model.evaluate(X_test, y_test)
        print(f"Final loss (MSE) on test set: {loss}")

        model.save(model_path)
        with open(scaler_path, "wb") as f:
            pickle.dump((feature_scaler, target_scaler), f)

        if return_predictions:
            # y_pred = model.predict(X_test)
            # test_dataset = tf.data.Dataset.from_tensor_slices(X_test.astype('float32')).batch(batch_size)
            # y_pred = model.predict(test_dataset)
            y_pred = model.predict(X_test, batch_size=batch_size)
            y_pred = target_scaler.inverse_transform(y_pred)
            y_true = target_scaler.inverse_transform(y_test)
            del features, features_scaled, target_scaled, df
            gc.collect()
            # all_objects = muppy.get_objects()
            # sum1 = summary.summarize(all_objects)
            # summary.print_(sum1)
            return model,history,batch_size, feature_scaler, target_scaler, y_true, y_pred
        return model,history,batch_size, feature_scaler, target_scaler




    def train_model_experiment_old(
        self,
        table_name: Coins,
        time_frame: Timeframe,
        limit: int = 1000000,
        window_size: int = 60,
        horizon: int = 1,
        model_path=None,
        scaler_path=None,
        return_predictions=False,
        epochs: int = 50,
        learning_rate: float = 0.001,
        dropout: float = 0.2,
        neyro: int = 64,
        df_ready=None,
        offset=None,
        batch_size=64
    ):
        # tf.debugging.set_log_device_placement(True)  # Включаем логирование размещения операций
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[0], 'GPU')
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
                )
                # tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError as e:
                print("Ошибка при настройке GPU:", e)

        

        if df_ready is None:
            # Загрузка данных из БД (ваш код)
            # ...
            pass
        else:
            df = df_ready.copy()
            df = df.drop(columns=["offset"])

        df["target"] = df["c"].shift(-horizon)
        df.dropna(inplace=True)

        features = df.drop(columns=["ts", "target"])

        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        features_scaled = feature_scaler.fit_transform(features)
        target_scaled = target_scaler.fit_transform(df[["target"]])

        def create_sequences(X, y, window_size, horizon):
            Xs, ys = [], []
            for i in range(len(X) - window_size - horizon + 1):
                Xs.append(X[i : (i + window_size)])
                ys.append(y[i + window_size + horizon - 1])
            return np.array(Xs), np.array(ys)

        X_lstm, y_lstm = create_sequences(features_scaled, target_scaled, window_size, horizon)

        split_idx = int(0.8 * len(X_lstm))
        X_train, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
        y_train, y_test = y_lstm[:split_idx], y_lstm[split_idx:]

        # Приводим к float32
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_test = y_test.astype(np.float32)

        # Создаём tf.data.Dataset с правильным порядком вызовов
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        # train_dataset = train_dataset.cache()  # кешируем исходные данные
        # train_dataset = train_dataset.shuffle(buffer_size=1024)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        # # Создаём модель с implementation=1 для cuDNN
        # model = tf.keras.Sequential([
        #     tf.keras.layers.LSTM(neyro, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), implementation=1),
        #     tf.keras.layers.LSTM(neyro // 2, return_sequences=False, implementation=1),
        #     tf.keras.layers.Dropout(dropout),
        #     tf.keras.layers.Dense(32, activation="relu"),
        #     tf.keras.layers.Dense(1)
        # ])
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(neyro, input_shape=(X_train.shape[1], X_train.shape[2]), implementation=1),
            tf.keras.layers.Dense(1)
        ])


        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")

        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=[early_stop],
            verbose=1,
        )

        loss = model.evaluate(val_dataset)
        print(f"Final loss (MSE) on test set: {loss}")

        model.save(model_path)
        with open(scaler_path, "wb") as f:
            import pickle
            pickle.dump((feature_scaler, target_scaler), f)

        if return_predictions:
            y_pred = model.predict(X_test)
            y_pred = target_scaler.inverse_transform(y_pred)
            y_true = target_scaler.inverse_transform(y_test)
            return model, history, batch_size, feature_scaler, target_scaler, y_true, y_pred
        return model, history, batch_size, feature_scaler, target_scaler


    
    
    


    def train_model_experiment_where_predictions(
        self,
        table_name: Coins,
        time_frame: Timeframe,
        limit: int = 500000,
        window_size: int = 60,
        horizon: int = 1,
        model_path=None,
        scaler_path=None,
        return_predictions=False,
        epochs: int = 50,
        learning_rate: float = 0.001,
        dropout: float = 0.2,
        neyro: int = 64,
        csv_path: str = "predictions.csv",  # путь для CSV
    ):

        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dropout, Dense
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        import pickle
        import os

        # Шаг 1: Загрузка данных
        query = f"""
            SELECT ts, o, h, l, c,
                vol, volCcy, volCcyQuote,
                ma50, ma200, ema12, ema26,
                macd, macd_signal, rsi14, macd_histogram,
                stochastic_k, stochastic_d
            FROM {table_name.value}
            WHERE timeFrame=%s AND ts >= %s
            ORDER BY ts ASC
            LIMIT %s;
        """
        params = (time_frame.label, 0, limit)
        rows = self.db.query_to_bd(query, params)

        columns = [
            "ts",
            "o",
            "h",
            "l",
            "c",
            "vol",
            "volCcy",
            "volCcyQuote",
            "ma50",
            "ma200",
            "ema12",
            "ema26",
            "macd",
            "macd_signal",
            "rsi14",
            "macd_histogram",
            "stochastic_k",
            "stochastic_d",
        ]

        df = pd.DataFrame(rows, columns=columns)

        # Шаг 2: Обработка
        df["target"] = df["c"].shift(-horizon)
        df.dropna(inplace=True)

        # Отделяем последние 30 строк (они будут использоваться после обучения)
        df_future = df.iloc[-30:].copy()
        df_train = df.iloc[:-30].copy()

        # Масштабируем
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        features_scaled = feature_scaler.fit_transform(
            df_train.drop(columns=["ts", "target"])
        )
        target_scaled = target_scaler.fit_transform(df_train[["target"]])

        # Формируем входные последовательности
        def create_sequences(X, y, window_size, horizon):
            Xs, ys = [], []
            for i in range(len(X) - window_size - horizon + 1):
                Xs.append(X[i : i + window_size])
                ys.append(y[i + window_size + horizon - 1])
            return np.array(Xs), np.array(ys)

        X_lstm, y_lstm = create_sequences(
            features_scaled, target_scaled, window_size, horizon
        )

        # Тренировочные и валидационные выборки
        split_idx = int(0.8 * len(X_lstm))
        X_train, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
        y_train, y_test = y_lstm[:split_idx], y_lstm[split_idx:]

        # Шаг 3: Обучение
        model = Sequential(
            [
                LSTM(
                    neyro,
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    return_sequences=False,
                ),
                Dropout(dropout),
                Dense(32, activation="relu"),
                Dense(1),
            ]
        )

        model.compile(optimizer=Adam(learning_rate), loss="mse")
        early_stop = EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=1,
        )

        loss = model.evaluate(X_test, y_test)
        print(f"Final loss (MSE) on test set: {loss}")

        # Сохраняем модель и скейлеры
        if model_path:
            model.save(model_path)
        if scaler_path:
            with open(scaler_path, "wb") as f:
                pickle.dump((feature_scaler, target_scaler), f)

        # Шаг 4: Прогноз на 30 будущих значений
        if return_predictions:
            # Берём последние window_size строк из обучающего фрейма + 30 новых
            full_df = pd.concat(
                [df_train.tail(window_size), df_future], ignore_index=True
            )
            features_all = feature_scaler.transform(
                full_df.drop(columns=["ts", "target"])
            )
            dummy_target = target_scaler.transform(
                full_df[["target"]]
            )  # чтобы не ломалось в create_sequences

            X_pred, y_true = create_sequences(
                features_all, dummy_target, window_size, horizon
            )

            y_pred = model.predict(X_pred)
            y_pred_inv = target_scaler.inverse_transform(y_pred)
            y_true_inv = target_scaler.inverse_transform(y_true)

            # Сохраняем в CSV
            pred_df = pd.DataFrame(
                {
                    "ts": df_future["ts"].values[: len(y_pred_inv)],
                    "real": y_true_inv.flatten(),
                    "predicted": y_pred_inv.flatten(),
                }
            )

            pred_df.to_csv(csv_path, index=False)
            print(f"Сохранено предсказание на реальные данные в: {csv_path}")

            return model, feature_scaler, target_scaler, y_true_inv, y_pred_inv

        return model, feature_scaler, target_scaler



    def load_model_and_scalers(self):
        self.model = load_model("lstm_model.h5", compile=False)
        self.feature_scaler = joblib.load("feature_scaler.save")
        self.target_scaler = joblib.load("target_scaler.save")
        print("Модель и скейлеры успешно загружены.")

    def predict_price(self, table_name: Coins, time_frame: Timeframe) -> float:
        if (
            self.model is None
            or self.feature_scaler is None
            or self.target_scaler is None
        ):
            raise ValueError(
                "Модель и скейлеры не загружены. Вызови load_model_and_scalers сначала."
            )

        query = f"""
            SELECT o, h, l, c,
                vol, volCcy, volCcyQuote,
                ma50, ma200, ema12, ema26,
                macd, macd_signal, rsi14, macd_histogram,
                stochastic_k, stochastic_d
            FROM {table_name.value}
            WHERE timeFrame = %s
            ORDER BY ts DESC
            LIMIT 60;
        """
        params = (time_frame.label,)
        rows = self.db.query_to_bd(query, params)

        if len(rows) < 60:
            raise ValueError(f"Недостаточно данных: {len(rows)}")

        keys = [
            "o",
            "h",
            "l",
            "c",
            "vol",
            "volCcy",
            "volCcyQuote",
            "ma50",
            "ma200",
            "ema12",
            "ema26",
            "macd",
            "macd_signal",
            "rsi14",
            "macd_histogram",
            "stochastic_k",
            "stochastic_d",
        ]

        input_array = np.array([[row[key] for key in keys] for row in rows[::-1]])

        if input_array.shape != (60, len(keys)):
            raise ValueError(f"Неверная форма массива: {input_array.shape}")

        scaled_input = self.feature_scaler.transform(input_array)
        X = np.expand_dims(scaled_input, axis=0)

        pred_scaled = self.model.predict(X)
        predicted_price = self.target_scaler.inverse_transform(pred_scaled)

        return float(predicted_price[0][0])

    def train_model(
        self,
        table_name: Coins,
        time_frame: Timeframe,
        limit: int = 500000,
        window_size: int = 60,
        horizon: int = 1,
    ):

        query = f""" SELECT ts, o, h, l, c,
                vol, volCcy, volCcyQuote,
                ma50, ma200, ema12, ema26,
                macd, macd_signal, rsi14, macd_histogram,
                stochastic_k, stochastic_d
            FROM {table_name.value}
                    where timeFrame=%s and ts>=%s
                    ORDER BY ts ASC
                    limit %s;
            """
        params = (time_frame.label, 0, limit)
        rows = self.db.query_to_bd(query, params)

        columns = [
            "ts",
            "o",
            "h",
            "l",
            "c",
            "vol",
            "volCcy",
            "volCcyQuote",
            "ma50",
            "ma200",
            "ema12",
            "ema26",
            "macd",
            "macd_signal",
            "rsi14",
            "macd_histogram",
            "stochastic_k",
            "stochastic_d",
        ]

        df = pd.DataFrame(rows, columns=columns)

        # Целевая — следующая цена закрытия
        df["target"] = df["c"].shift(-horizon)
        df.dropna(inplace=True)

        features = df.drop(columns=["ts", "target"])

        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        features_scaled = feature_scaler.fit_transform(features)
        target_scaled = target_scaler.fit_transform(df[["target"]])

        def create_sequences(X, y, window_size, horizon):
            Xs, ys = [], []
            for i in range(len(X) - window_size - horizon + 1):
                Xs.append(X[i : (i + window_size)])
                ys.append(y[i + window_size + horizon - 1])
            return np.array(Xs), np.array(ys)

        # window_size = 60
        X_lstm, y_lstm = create_sequences(
            features_scaled, target_scaled, window_size, horizon
        )

        split_idx = int(0.8 * len(X_lstm))
        X_train, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
        y_train, y_test = y_lstm[:split_idx], y_lstm[split_idx:]

        model = Sequential(
            [
                LSTM(
                    64,
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    return_sequences=False,
                ),
                Dense(32, activation="relu"),
                Dense(1),
            ]
        )

        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

        early_stop = EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )

        history = model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=1,
        )

        loss = model.evaluate(X_test, y_test)
        print(f"Final loss (MSE) on test set: {loss}")

        model.save("lstm_model.h5")

        # Визуализация прогресса обучения
        plt.figure(figsize=(8, 5))
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.legend()
        plt.grid(True)
        plt.show()

        joblib.dump(feature_scaler, "feature_scaler.save")
        joblib.dump(target_scaler, "target_scaler.save")

        return model, feature_scaler, target_scaler
