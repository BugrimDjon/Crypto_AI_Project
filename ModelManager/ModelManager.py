import os
import re
import joblib
import numpy as np
from collections import defaultdict
from tensorflow.keras.models import load_model
import pandas as pd


class ModelManager:
    def __init__(self, base_dir="./top_models2"):
        """
        Инициализация класса — задаём путь до папок с моделями и скейлерами.
        """
        self.models_dir = os.path.join(base_dir, "models")
        self.scalers_dir = os.path.join(base_dir, "scalers")
        self.models_info = self._load_model_scaler_pairs()

        # Кэш для уже загруженных моделей и скейлеров, чтобы не загружать их заново каждый раз
        self.loaded_models = {}
        self.loaded_scalers = {}

    def list_models_out_df(self):
        """
        Возвращает DataFrame с параметрами всех моделей, включая пути к моделям и скейлерам.
        """
        rows = []
        for info in self.models_info.values():
            row = info["params"].copy()
            row["model_path"] = info.get("model_path")
            row["scaler_path"] = info.get("scaler_path")
            rows.append(row)
        return pd.DataFrame(rows)

    def _extract_model_params(self, filename):
        """
        Внутренний метод — извлекает параметры из имени файла.
        Используется регулярное выражение для разбора имени.
        """
        pattern = r"(?P<tf>_(1min|5min|10min|15min|30min|1hour|4hour|1day|1week))_ws(?P<ws>\d+)_hz(?P<hz>\d+)_le_ra(?P<ra>[\d.]+)_dr(?P<dr>[\d.]+)_ney(?P<ney>\d+)(?:_offset(?P<offset>\d+))?"

        match = re.search(pattern, filename)
        if not match:
            return None
        return {
            "filename": filename,
            "key": match.group(0),  # используется как уникальный идентификатор
            "timeframe": match.group("tf"),
            "window_size": int(match.group("ws")),
            "horizon": int(match.group("hz")),
            "learning_rate": float(match.group("ra")),
            "dropout": float(match.group("dr")),
            "neurons": int(match.group("ney")),
            "offset": int(match.group("offset")) if match.group("offset") is not None else 0,
        }

    def _load_model_scaler_pairs(self):
        """
        Внутренний метод — сопоставляет все модели и скейлеры по ключу.
        Возвращает словарь с полными путями и параметрами.
        """
        model_dict = {}

        # Проходим по всем моделям
        for filename in os.listdir(self.models_dir):
            if filename.endswith(".keras") or filename.endswith(".h5"):
                params = self._extract_model_params(filename)
                if params:
                    key = params["key"]
                    model_dict[key] = {
                        "model_path": os.path.join(self.models_dir, filename),
                        "params": params,
                    }

        # Добавляем к ним скейлеры
        for filename in os.listdir(self.scalers_dir):
            if filename.endswith(".pkl"):
                params = self._extract_model_params(filename)
                if params:
                    key = params["key"]
                    if key in model_dict:
                        model_dict[key]["scaler_path"] = os.path.join(
                            self.scalers_dir, filename
                        )

        return model_dict

    def list_models(self):
        """
        Возвращает список всех доступных моделей и их параметров.
        """
        return [info["params"] for info in self.models_info.values()]

    def get_model_by_params(self, timeframe, window_size, horizon, filename):
        """
        Возвращает путь к модели и скейлеру по заданным параметрам.
        """
        for info in self.models_info.values():
            p = info["params"]
            if (
                p["timeframe"] == timeframe
                and p["window_size"] == window_size
                and p["horizon"] == horizon
                and p["filename"] == filename
            ):
                return info
        return None

    def load_model_and_scaler(self, timeframe, window_size, horizon, filename):
        """
        Загружает и возвращает модель и скейлер по заданным параметрам.
        """
        info = self.get_model_by_params(timeframe, window_size, horizon, filename)
        if not info:
            raise ValueError("Модель с указанными параметрами не найдена.")

        model = load_model(info["model_path"], compile=False)
        feature_scaler, target_scaler = joblib.load(info["scaler_path"])
        print(info["model_path"])
        return model, feature_scaler, target_scaler

    def predict(self, X_input, timeframe, window_size, horizon, filename):
        """
        Выполняет прогноз по входному массиву `X_input` с использованием модели и скейлера,
        подобранных по параметрам `timeframe`, `window_size`, `horizon`.
        """
        model, feature_scaler, target_scaler = self.load_model_and_scaler(
            timeframe, window_size, horizon, filename
        )

        # Нормализация входных данных
        X_scaled = feature_scaler.transform(X_input)

        # Добавляем размерность, если вход одномерный
        if X_scaled.ndim == 2:
            X_scaled = np.expand_dims(X_scaled, axis=0)

        # Предсказание
        y_pred = model.predict(X_scaled)

        # Обратное преобразование, если требуется (по желанию)
        y_pred = target_scaler.inverse_transform(y_pred)

        return y_pred


    def predict_on_working_models(self, X_input, model_path, scaler_path):
        """
        Выполняет прогноз по входному массиву `X_input` с использованием модели и скейлера.
        Кэширует загруженные модели и скейлеры, чтобы не загружать их повторно.

        Параметры:
            X_input (np.ndarray): Входной массив признаков (последовательность для модели).
            model_path (str): Путь к .keras или .h5 файлу модели.
            scaler_path (str): Путь к .pkl файлу с двумя скейлерами (feature и target).

        Возвращает:
            y_pred (np.ndarray): Прогноз модели, приведённый к исходному масштабу.
        """
        # Загружаем модель из кэша или с диска, если еще не загружена
        if model_path not in self.loaded_models:
            model = load_model(model_path, compile=False)
            self.loaded_models[model_path] = model
        else:
            model = self.loaded_models[model_path]

        # Загружаем скейлеры из кэша или с диска, если еще не загружены
        if scaler_path not in self.loaded_scalers:
            scalers = joblib.load(scaler_path)
            self.loaded_scalers[scaler_path] = scalers
        else:
            scalers = self.loaded_scalers[scaler_path]

        feature_scaler, target_scaler = scalers

        # Масштабируем входной массив (нормализация признаков)
        X_scaled = feature_scaler.transform(X_input)

        # Если подаётся 2D-массив (одна последовательность), добавим batch-ось
        if X_scaled.ndim == 2:
            X_scaled = np.expand_dims(X_scaled, axis=0)  # [1, window_size, num_features]

        # Прогноз модели
        y_pred = model.predict(X_scaled)

        # Обратное масштабирование целевой переменной
        y_pred = target_scaler.inverse_transform(y_pred)

        return y_pred
