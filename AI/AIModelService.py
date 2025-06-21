# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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
import pickle

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

    def train_model_experiment(
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
        learning_rate: float = 0.001,  # регулируем
        dropout: float = 0.2,  # регулируем
        neyro: int = 64,
        df_ready=None,
        offset=None,
        batch_size=64
    ):
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
        else:
            df=df_ready.copy()
            df = df.drop(columns=["offset"])




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

        # ---- Добавлено: перемешивание обучающей выборки ----
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
        # ------------------------------------------------------

        # model = Sequential(
        #     [
        #         LSTM(
        #             neyro,
        #             input_shape=(X_train.shape[1], X_train.shape[2]),
        #             return_sequences=False,
        #         ),
        #         Dropout(dropout),  # регулируем
        #         Dense(32, activation="relu"),
        #         Dense(1),
        #     ]
        # )
        # import os
        # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        # neyro=4
        model = Sequential([
        LSTM(neyro, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2]), implementation=1, unroll=True),
        Dropout(dropout),
        Dense(64, activation="relu"),  # <- новый слой
        Dense(32, activation="relu"),  # <- ещё один
        Dense(1)
        ])

        # print("LSTM implementation:", layer.implementation)
        model.compile(optimizer=Adam(learning_rate), loss="mse")

        early_stop = EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=1,
        )

        loss = model.evaluate(X_test, y_test)
        print(f"Final loss (MSE) on test set: {loss}")

        model.save(model_path)
        with open(scaler_path, "wb") as f:
            pickle.dump((feature_scaler, target_scaler), f)

        if return_predictions:
            y_pred = model.predict(X_test)
            y_pred = target_scaler.inverse_transform(y_pred)
            y_true = target_scaler.inverse_transform(y_test)
            return model,history,batch_size, feature_scaler, target_scaler, y_true, y_pred
        return model,history,batch_size, feature_scaler, target_scaler

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
