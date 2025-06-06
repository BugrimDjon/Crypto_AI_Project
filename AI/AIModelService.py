# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from tensorflow.keras.models import load_model, Sequential

from enums.coins import Coins
from database.db import Database
from services.okx_candles import OkxCandlesFetcher
from services.time_control import TimControl
from enums.timeframes import Timeframe
from config.SettingsCoins import SettingsCoins
from enums.AfterBefore import AfterBefore
from logger.context_logger import ContextLogger

import pandas as pd
import ta
import json
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,  # или DEBUG, WARNING, ERROR
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),  # Писать в файл
        logging.StreamHandler(),  # Писать в консоль
    ],
)


class AIModelService:
    def __init__(self, db: Database) -> None:
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.db = db

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
