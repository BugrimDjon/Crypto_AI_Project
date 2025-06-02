from enums.coins import Coins
from database.db import Database
from services.okx_candles import OkxCandlesFetcher
from services.time_control import TimControl
from enums.timeframes import Timeframe
from config.SettingsCoins import SettingsCoins
from enums.AfterBefore import AfterBefore
from logger.context_logger import ContextLogger

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
        self.db = db

    def first_load_candles(self, coin: Coins, length_in_hours=365 * 24):

        TimControl.frequency_limitation(10)
        fetcher = OkxCandlesFetcher(instId=coin, bar=Timeframe._1min)
        data = fetcher.fetch_candles(limit=60)
        for candle in data:
            candle["timeFrame"] = "1m"
            candle["quoteCoin"] = SettingsCoins.quote_coin().value

        self.db.insert_many_candles(data, coin.value)

        for x in range(length_in_hours):
            TimControl.frequency_limitation(10)
            data = fetcher.fetch_candles(
                limit=60, afterBefore=AfterBefore.After, start_time=data[-1]["ts"]
            )
            for candle in data:
                candle["timeFrame"] = "1m"
                candle["quoteCoin"] = SettingsCoins.quote_coin().value

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
        delta_candles = int((stop_time - start_time) / 1000) / time_frame.minutes
        while delta_candles > 0:
            if delta_candles > 10000:
                delta_candles = 10000
            step = time_frame.minutes * 1000 * 60
            query = f""" SELECT ts FROM {table_name}
                    where timeFrame=%s and ts>%s
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
                    fetcher = OkxCandlesFetcher(instId=table_name, bar=time_frame)
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
                        with open(
                            r"D:\Project_programming\for_AI\data_candles.json",
                            "w",
                            encoding="utf-8",
                        ) as f:
                            json.dump(data_candles, f, ensure_ascii=False, indent=4)
                        with open(
                            r"D:\Project_programming\for_AI\test.json",
                            "w",
                            encoding="utf-8",
                        ) as f:
                            json.dump(request, f, ensure_ascii=False, indent=4)
                        for x in data_candles:
                            x["timeFrame"] = time_frame.label
                            x["quoteCoin"] = SettingsCoins.quote_coin()

                        output_data["db errors"] = self.db.insert_many_candles(
                            data_candles, table_name, True
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

    def recalc_timeframe(self, baseCoin: Coins, from_tf: Timeframe, to_tf: Timeframe):
        interval_minutes = to_tf.minutes / from_tf.minutes
        if not interval_minutes.is_integer():
            print("Входные таймфреймы не совместимы")
            return
        interval_minutes = int(interval_minutes)
        # Получаем максимальный ts для нужного таймфрейма
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

        # Аггрегируем
        aggregated_candles = Database.aggregate_candles(candles, to_tf)

        # Вставляем обратно
        self.db.insert_many_candles(aggregated_candles, name_table=baseCoin.value)

        print(f"Пересчитано {len(aggregated_candles)} свечей с {from_tf} в {to_tf}")
        return len(aggregated_candles)
