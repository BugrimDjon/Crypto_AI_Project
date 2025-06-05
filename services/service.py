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


class Servise:
    def __init__(self, db: Database) -> None:
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.db = db

    def load_model_and_scalers(self):
        self.model = load_model('lstm_model.h5', compile=False)
        self.feature_scaler = joblib.load('feature_scaler.save')
        self.target_scaler = joblib.load('target_scaler.save')
        print("Модель и скейлеры успешно загружены.")

    def out_ai(self, table_name: Coins, time_frame: Timeframe) -> float:
        if self.model is None or self.feature_scaler is None or self.target_scaler is None:
            raise ValueError("Модель и скейлеры не загружены. Вызови load_model_and_scalers сначала.")

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

        keys = ['o', 'h', 'l', 'c',
                'vol', 'volCcy', 'volCcyQuote',
                'ma50', 'ma200', 'ema12', 'ema26',
                'macd', 'macd_signal', 'rsi14', 'macd_histogram',
                'stochastic_k', 'stochastic_d']

        input_array = np.array([[row[key] for key in keys] for row in rows[::-1]])

        if input_array.shape != (60, len(keys)):
            raise ValueError(f"Неверная форма массива: {input_array.shape}")

        scaled_input = self.feature_scaler.transform(input_array)
        X = np.expand_dims(scaled_input, axis=0)

        pred_scaled = self.model.predict(X)
        predicted_price = self.target_scaler.inverse_transform(pred_scaled)

        return float(predicted_price[0][0])







    def first_load_candles(self, coin: Coins, length_in_hours: int=365 * 24):

        TimControl.frequency_limitation(10)
        fetcher = OkxCandlesFetcher(instId=coin.value, bar=Timeframe._1min)
        data = fetcher.fetch_candles(limit=60)
        for candle in data:
            candle["timeFrame"] = "1m"
            candle["quoteCoin"] = SettingsCoins.quote_coin()

        self.db.insert_many_candles(data, coin.value)

        for x in range(int(length_in_hours)):
            TimControl.frequency_limitation(10)
            data = fetcher.fetch_candles(
                limit=60, afterBefore=AfterBefore.After, start_time=data[-1]["ts"]
            )
            for candle in data:
                candle["timeFrame"] = "1m"
                candle["quoteCoin"] = SettingsCoins.quote_coin()

            self.db.insert_many_candles(data, coin.value)

            print(
                f"Обработано {x*100/length_in_hours:.3f}%. ( {x*60} минут из {length_in_hours*60})"
            )

    def check_sequence_timeframes(
        self,
        table_name: Coins,
        time_frame: Timeframe,
        start_time: int,
        stop_time: int,
        return_result: bool = False,
    ):
        """
        Проверяет последовательность таймфреймов в таблице.
         Параметры:
            table_name (Coins): Название таблицы обязательно должно соответствовать названию монеты.
            time_frame (Timeframe): Таймфрейм (перечисления) (например, _1m, _5m).
            start_time (int): Время начала проверки в формате timestamp (миллисекунды).
            stop_time (int): Время окончания проверки в формате timestamp (миллисекунды).
            return_result (bool = False): необязательный параметр, указывающий надо ли возвращать ответ

        Возвращает:
            output_data = {
                "missing data": 0, - количество отсутствующих записей нарушабщих целосность
                "db errors": 0     - количество ошибок при записи в БД
            }

        """
        import inspect

        current_frame = inspect.currentframe()
        method_name = current_frame.f_code.co_name
        class_name = self.__class__.__name__

        output_data = {"missing data": 0, "db errors": 0}

        if start_time > stop_time:
            start_time, stop_time = stop_time, start_time
        delta_candles = int(((stop_time - start_time) / 1000) / time_frame.minutes)
        while delta_candles > 0:
            if delta_candles > 10000:
                delta_candles = 10000
            step = time_frame.minutes * 1000 * 60
            query = f""" SELECT ts FROM {table_name.value}
                    where timeFrame=%s and ts>=%s
                    ORDER BY ts ASC
                    limit %s;
            """
            params = (time_frame.label, start_time, delta_candles)
            # производим запрос в бд порцию записей
            request = self.db.query_to_bd(query, params)
            len_request = len(request)
            for i in range(len_request - 1):
                # если разница между соседними записями не соответствует step
                # значит есть пропуски
                if (int(request[i + 1]["ts"]) - int(request[i]["ts"])) > step:
                    number_of_passes = int(
                        (request[i + 1]["ts"] - request[i]["ts"]) / step
                    )
                    logging.info(
                        ContextLogger.string_context()
                        + f""": 
                                 Обнаружены пропуски в последовательности {time_frame.label} таймфрейма колличестве {number_of_passes}"""
                    )
                    output_data["missing data"] += number_of_passes
                    start_time_for_reqwest = int(request[i + 1]["ts"])
                    fetcher = OkxCandlesFetcher(instId=table_name.value, bar=time_frame)
                    while number_of_passes > 0:
                        if number_of_passes > 100:
                            number_of_candles = 100
                            limit = number_of_candles
                        else:
                            number_of_candles = number_of_passes
                            limit = number_of_candles - 1
                        
                        data_candles = fetcher.fetch_candles(
                            limit=limit,
                            afterBefore=AfterBefore.After,
                            start_time=str(start_time_for_reqwest),
                        )
                        # with open(
                        #     r"D:\Project_programming\for_AI\data_candles.json",
                        #     "w",
                        #     encoding="utf-8",
                        # ) as f:
                        #     json.dump(data_candles, f, ensure_ascii=False, indent=4)
                        # with open(
                        #     r"D:\Project_programming\for_AI\test.json",
                        #     "w",
                        #     encoding="utf-8",
                        # ) as f:
                        #     json.dump(request, f, ensure_ascii=False, indent=4)
                        for x in data_candles:
                            x["timeFrame"] = time_frame.label
                            x["quoteCoin"] = SettingsCoins.quote_coin()

                        output_data["db errors"] = self.db.insert_many_candles(
                            data_candles, table_name.value, True
                        )
                        # start_time_for_reqwest=start_time_for_reqwest+number_of_candles
                        start_time_for_reqwest -= int(
                            (number_of_candles) * time_frame.minutes * 60 * 1000
                        )
                        number_of_passes = number_of_passes - number_of_candles

            start_time = int(request[-1]["ts"])
            delta_candles = ((stop_time - start_time) / 1000) / time_frame.minutes

        if return_result:
            return output_data

    def calculation_of_indicators(self, tableName: Coins):
        """
        Выполняет расчет технических индикаторов (MA, EMA, MACD, RSI, Stochastic)
        для заданной таблицы и всех таймфреймов, указанных в перечислении Timeframe.

        Алгоритм работы:
        1. Определяет последнее рассчитанное время (`MAX(ts)`), где уже посчитан индикатор `ma200`.
        2. Отступает назад на 300 свечей, чтобы пересчитать "хвост" с запасом (перекрытие).
        3. Загружает из базы данных порцию данных с указанного времени (`start_ts`) и таймфрейма.
        4. Преобразует записи в DataFrame, рассчитывает индикаторы:
        - MA50, MA200
        - EMA12, EMA26
        - MACD, MACD Signal, MACD Histogram
        - RSI14
        - Stochastic %K и %D
        5. Если используется перекрытие (данные уже частично рассчитаны),
        то обрезаются первые 300 строк, чтобы избежать повторной перезаписи.
        6. Преобразует результат обратно в список словарей и сохраняет в базу данных.
        7. Цикл повторяется до тех пор, пока не будут обработаны все доступные данные.

        Параметры:
            tableName (Coins): Перечисление с именем таблицы для обработки.

        Примечания:
            - Для расчетов требуется минимум 200 строк.
            - Используется ограничение на количество строк (limit = 10000) при загрузке из БД.
            - Повторный расчет последних 300 строк обеспечивает непрерывность индикаторов.
            - Данные сохраняются по мере продвижения вперед по времени на основе MAX(ts).
        """
        limit = 10000
        for tm in Timeframe:
            lastLap = False
            old_ts = 0
            counter = 0
            number_of_records = 0

            while not lastLap:
                # Получаем последнее время, где уже рассчитан ma200
                query = f"""
                    SELECT MAX(ts) AS max_ts FROM {tableName.value}
                    WHERE ma200 IS NOT NULL AND timeFrame = %s
                """
                params = (tm.label,)
                result = self.db.query_to_bd(query, params)
                last_calculated_ts = (
                    result[0]["max_ts"] if result and result[0]["max_ts"] else 0
                )

                # Отступаем назад на 300 свечей, чтобы пересчитать хвост (если надо)
                start_ts = int(last_calculated_ts) - tm.minutes * 60 * 1000 * 300
                if start_ts < 0:
                    start_ts = 0

                # Получаем следующую порцию данных
                query = f"""
                    SELECT * FROM {tableName.value}
                    WHERE ts >= %s AND timeFrame = %s
                    ORDER BY ts
                    LIMIT %s
                """
                params = (start_ts, tm.label, limit)
                records = self.db.query_to_bd(query, params)

                if not records:
                    break  # нечего обрабатывать

                if (len(records) < limit) or (old_ts == records[0]["ts"]):
                    lastLap = True

                old_ts = records[0]["ts"]
                # Преобразуем список словарей в DataFrame
                df = pd.DataFrame(records)

                # Если строк меньше 200, индикаторы считать нельзя
                if len(df) < 200:
                    print(
                        f"Недостаточно данных для расчета некоторых индикаторов по {tm.label}"
                    )
                    # continue

                # Расчёт индикаторов
                df["ma50"] = ta.trend.sma_indicator(df["c"], window=50)
                df["ma200"] = ta.trend.sma_indicator(df["c"], window=200)
                df["ema12"] = ta.trend.ema_indicator(df["c"], window=12)
                df["ema26"] = ta.trend.ema_indicator(df["c"], window=26)
                df["macd"] = ta.trend.macd(df["c"])
                df["macd_signal"] = ta.trend.macd_signal(df["c"])
                df["macd_histogram"] = df["macd"] - df["macd_signal"]
                df["rsi14"] = ta.momentum.rsi(df["c"], window=14)

                stoch = ta.momentum.StochasticOscillator(
                    high=df["h"], low=df["l"], close=df["c"], window=14, smooth_window=3
                )
                df["stochastic_k"] = stoch.stoch()
                df["stochastic_d"] = stoch.stoch_signal()

                if start_ts > 0:
                    # Обрезаем первые 300 строк
                    df = df.iloc[300:]
                    if df.empty:
                        continue

                # Преобразуем обратно в список словарей
                result_to_save = [
                    {k: (None if pd.isna(v) else v) for k, v in row.items()}
                    for row in df.to_dict(orient="records")
                ]

                counter += 1
                number_of_records += len(df)
                logging.info(
                    ContextLogger.string_context()
                    + f"""
                Таймфрейм = {tm.label},        проход - {counter},          обработано - {number_of_records} записей"""
                )
                # Сохраняем
                self.db.insert_many_indikator(result_to_save, tableName.value)

    def recalc_timeframe(self, baseCoin: Coins, from_tf: Timeframe, to_tf: Timeframe):
        end_cikle: bool=False
        caunter=0
        number=0
        while end_cikle==False:
            interval_minutes = to_tf.minutes / from_tf.minutes
            if not interval_minutes.is_integer():
                print("Входные таймфреймы не совместимы")
                return
            interval_minutes = int(interval_minutes)
            # Получаем максимальный ts для нужного таймфрейма, самый свежий ts когда был рассчитан данный таймфрейм
            last_ts = self.db.get_max_timestamp(
                baseCoin.value, timeFrame=to_tf.label, quoteCoin=SettingsCoins.quote_coin()
            )

            if last_ts is None:
                # Если данных нет, возьмём минимум по from_tf
                candles = self.db.fetch_candles_from_ts(
                    baseCoin.value, from_tf.label, 0, SettingsCoins.quote_coin(), 10000
                )
            else:
                # Начинаем с ts следующей свечи
                start_ts = last_ts + interval_minutes * 60 * 1000
                candles = self.db.fetch_candles_from_ts(
                    baseCoin.value,
                    from_tf.label,
                    start_ts,
                    SettingsCoins.quote_coin(),
                    10000,
                )

                # Если len(candles) то свеча однозначно не полная и пересчитывать не надо
                # это модет быть когда пересчитал 15 минутный таймфрейм в 15 (30) мин. и в 15 (30) мин
                # опрашиваеш снова, тогда форминуется len(candles)>1 а так как метод aggregate_candles
                # требует как минимум 2 списка в aggregate_candles то это приводит к ошибке
            if len(candles) > 1:
                # Аггрегируем
                aggregated_candles = Database.aggregate_candles(candles, to_tf)

                # Вставляем обратно
                self.db.insert_many_candles(aggregated_candles, name_table=baseCoin.value)
                number+=1
                print("Проход - "+str(number))
                caunter +=len(aggregated_candles)
                if (len(aggregated_candles)==0):
                    print(f"Пересчитано {caunter} свечей с {from_tf} в {to_tf}")
                    return caunter

            else:
                print(f"Пересчитано {caunter} свечей с {from_tf} в {to_tf}")
                end_cikle=True
                return caunter
            

    def data_for_update(self, table_name=Coins):
        query = f"""
                    SELECT MAX(ts) AS max_ts FROM {table_name.value}
                    WHERE timeFrame= "1m"
                """
        result = self.db.query_to_bd(query, ())
        last_time_in_database = result[0]["max_ts"]

        fetcher = OkxCandlesFetcher(instId=table_name.value, bar=Timeframe._1min)
        data = fetcher.fetch_candles(limit=1)

        if last_time_in_database < int(data[0]["ts"]):
            for x in data:
                x["timeFrame"] = Timeframe._1min.label
                x["quoteCoin"] = SettingsCoins.quote_coin()

            self.db.insert_many_candles(data, table_name.value)
        query = f"""
                    SELECT ts FROM {table_name.value} 
                    where ma200 is not null and 
                    timeFrame='1m' 
                    order by ts DESC 
                    limit 1;
                    
                """
        result = self.db.query_to_bd(query, ())
        time_in_database = result[0]["ts"]

        return {
            "time_in_database": time_in_database,
            "current_time_on_the_exchange": data[0]["ts"],
        }

    
    def for_ai(self, table_name: Coins, time_frame: Timeframe, limit: int=500000, window_size:int=60, horizon:int=1):

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
            'ts', 'o', 'h', 'l', 'c',
            'vol', 'volCcy', 'volCcyQuote',
            'ma50', 'ma200', 'ema12', 'ema26',
            'macd', 'macd_signal', 'rsi14', 'macd_histogram',
            'stochastic_k', 'stochastic_d'
        ]

        df = pd.DataFrame(rows, columns=columns)

        # Целевая — следующая цена закрытия
        df['target'] = df['c'].shift(-horizon)
        df.dropna(inplace=True)

        features = df.drop(columns=['ts', 'target'])

        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        features_scaled = feature_scaler.fit_transform(features)
        target_scaled = target_scaler.fit_transform(df[['target']])

        def create_sequences(X, y, window_size, horizon):
            Xs, ys = [], []
            for i in range(len(X) - window_size - horizon + 1):
                Xs.append(X[i:(i + window_size)])
                ys.append(y[i + window_size + horizon - 1])
            return np.array(Xs), np.array(ys)

        # window_size = 60
        X_lstm, y_lstm = create_sequences(features_scaled, target_scaled, window_size, horizon)

        split_idx = int(0.8 * len(X_lstm))
        X_train, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
        y_train, y_test = y_lstm[:split_idx], y_lstm[split_idx:]

        model = Sequential([
            LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=1
        )

        loss = model.evaluate(X_test, y_test)
        print(f'Final loss (MSE) on test set: {loss}')

        model.save('lstm_model.h5')

        # Визуализация прогресса обучения
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.legend()
        plt.grid(True)
        plt.show()

        joblib.dump(feature_scaler, 'feature_scaler.save')
        joblib.dump(target_scaler, 'target_scaler.save')

        return model, feature_scaler, target_scaler
    

#     from tensorflow.keras.models import load_model
# import joblib



