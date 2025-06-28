# -*- coding: utf-8 -*-

import inspect
import mysql.connector
from mysql.connector import Error
from config import settings
from collections import defaultdict
from enums.timeframes import Timeframe
from config.SettingsCoins import SettingsCoins
from logger.context_logger import ContextLogger
import platform
import subprocess
import re

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


class Database:
    def __init__(self):
        self.connection = None

    @staticmethod
    def get_windows_ip():
        """
        Получить IPv4-адрес Windows из WSL, вызывая PowerShell.
        Игнорирует IP из диапазона APIPA (169.254.x.x) и localhost (127.0.0.1).
        Возвращает первый подходящий IP или "127.0.0.1" если не найден.
        """
        try:
            if platform.system() == "Linux":
                cmd = [
                    "powershell.exe",
                    "-Command",
                    "Get-NetIPAddress -AddressFamily IPv4 | "
                    "Where-Object { $_.IPAddress -notlike '169.254.*' -and $_.IPAddress -ne '127.0.0.1' -and $_.InterfaceAlias -notlike '*WSL*' } | "
                    "Select-Object -ExpandProperty IPAddress"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                ips = result.stdout.strip().splitlines()
                for ip in ips:
                    if ip and not ip.startswith("169.254.") and ip != "127.0.0.1":
                        return ip
                return "127.0.0.1"
            elif platform.system() == "Windows":
                return "127.0.0.1"
            else:
                return "127.0.0.1"
        except Exception as e:
            print(f"Ошибка при получении IP Windows: {e}")
            return "127.0.0.1"

    @staticmethod
    def get_db_host():
        system = platform.system()
        if system == "Linux":
            return Database.get_windows_ip()
        elif system == "Windows":
            return "127.0.0.1"
        else:
            return "127.0.0.1"

    def connect(self):
        try:
            host = self.get_db_host()
            self.connection = mysql.connector.connect(
                host=host,
                port=settings.DB_PORT,
                user=settings.DB_USER,
                password=settings.DB_PASSWORD,
                database=settings.DB_NAME,
            )
            if self.connection.is_connected():
                print(f"Успешное подключение к базе данных на хосте {host}")
        except Error as e:
            print(f"Ошибка подключения к базе: {e}")


    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("🔌 Соединение с базой закрыто")

    def insert_candle(
        self,
        ts,
        o,
        h,
        l,
        c,
        vol,
        volCcy,
        volCcyQuote,
        confirm,
        timeFrame,
        quoteCoin,
        ma50=None,
        ma200=None,
        ema12=None,
        ema26=None,
        macd=None,
        macd_signal=None,
        macd_histogram=None,
        rsi14=None,
        stochastic_k=None,
        stochastic_d=None,
    ):

        if not self.connection or not self.connection.is_connected():
            print("⚠️ Нет соединения с базой данных")
            return
        query = """
        INSERT INTO fet_data 
        (ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm, timeFrame, quoteCoin, 
         ma50, ma200, ema12, ema26, macd, macd_signal, macd_histogram, rsi14, stochastic_k, stochastic_d)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            ts,
            o,
            h,
            l,
            c,
            vol,
            volCcy,
            volCcyQuote,
            confirm,
            timeFrame,
            quoteCoin,
            ma50,
            ma200,
            ema12,
            ema26,
            macd,
            macd_signal,
            macd_histogram,
            rsi14,
            stochastic_k,
            stochastic_d,
        )
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, values)
            self.connection.commit()
            # print(f"✅ Запись вставлена: ts={ts}")
        except Error as e:
            print(f"❌ Ошибка вставки данных: {e}")

    def fetch_candles(self, limit=10):
        query = "SELECT * FROM fet_data ORDER BY ts DESC LIMIT %s"
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, (limit,))
            return cursor.fetchall()
        except Error as e:
            print(f"❌ Ошибка чтения данных: {e}")
            return []

    def query_to_bd(self, query: str, params: tuple = ()):
        """
        метод универсальный для запроса в БД
        параметры
            query (str): SQL-запрос с плейсхолдерами %s.
            params (tuple): пречень параметров для запроса, подставляется
                            последовательно вместо "%s" в запрос
        возвращает мвссив
        """
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params)
            result = cursor.fetchall()
            cursor.close()
            return result
        except Error as e:
            logging.error(
                f"Ошибка при выполнении запроса:\n{query}\nПараметры: {params}\nОшибка: {e}"
            )
            return []

    def insert_many_candles(
        self, candle_list, name_table: str, return_result: bool = False
    ):
        output_data = 0
        if not self.connection or not self.connection.is_connected():
            print("⚠️ Нет соединения с базой данных")
            if return_result:
                output_data += 1
                return output_data
            return

        query = f"""
        INSERT INTO {name_table}
        (ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm, timeFrame, quoteCoin, 
         ma50, ma200, ema12, ema26, macd, macd_signal, macd_histogram, rsi14, stochastic_k, stochastic_d)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        # print(query)
        values = []
        for temp in candle_list:
            values.append(
                (
                    temp["ts"],
                    temp["o"],
                    temp["h"],
                    temp["l"],
                    temp["c"],
                    temp["vol"],
                    temp["volCcy"],
                    temp["volCcyQuote"],
                    temp["confirm"],
                    temp["timeFrame"],
                    temp["quoteCoin"],
                    temp.get("ma50"),
                    temp.get("ma200"),
                    temp.get("ema12"),
                    temp.get("ema26"),
                    temp.get("macd"),
                    temp.get("macd_signal"),
                    temp.get("macd_histogram"),
                    temp.get("rsi14"),
                    temp.get("stochastic_k"),
                    temp.get("stochastic_d"),
                )
            )

        try:
            cursor = self.connection.cursor()
            cursor.executemany(query, values)
            self.connection.commit()
            # print(f"✅ Вставлено записей: {len(values)}")
        except Error as e:
            output_data += 1
            print(f"❌ Ошибка при пакетной вставке: {e}")
            if return_result:
                return output_data

    #  Функция для получения максимального ts по таймфрейму и базе (в классе Database)
    def get_max_timestamp(self, tableName: str, timeFrame: str, quoteCoin: str):
        try:
            # ⚠️ Проверяем, что имя таблицы безопасно — латиница, цифры, подчёркивания
            if not tableName.isidentifier():
                raise ValueError(f"Недопустимое имя таблицы: {tableName}")

            query = f"""
            SELECT MAX(ts) as max_ts FROM `{tableName}`
            WHERE timeFrame = %s AND quoteCoin = %s
            """
            # print(query,timeFrame ,quoteCoin)
            cursor = self.connection.cursor()
            cursor.execute(query, (timeFrame, quoteCoin))
            result = cursor.fetchone()
            return result[0] if result and result[0] is not None else None
        except Error as e:
            print(f"Ошибка при получении max(ts): {e}")
            return None

    # 2. Функция выборки 1m свечей с определённого времени (расширим уже имеющуюся fetch_candles)
    def fetch_candles_from_ts(
        self,
        tableName: str,
        timeFrame: str,
        from_ts: int,
        quoteCoin: str,
        batch_size: int = 1000,
    ):
        try:
            # ⚠️ Проверяем, что имя таблицы безопасно — латиница, цифры, подчёркивания
            if not tableName.isidentifier():
                raise ValueError(f"Недопустимое имя таблицы: {tableName}")

            query = f"""
            SELECT * FROM {tableName}
            WHERE timeFrame = %s AND quoteCoin = %s AND ts >= %s
            ORDER BY ts ASC
            LIMIT {batch_size}
            """
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, (timeFrame, quoteCoin, from_ts))
            return cursor.fetchall()
        except Error as e:
            print(f"Ошибка чтения данных: {e}")
            return []

    # 3. Функция агрегирования свечей в N-минутные свечи (например, 5m)
    @staticmethod
    # def aggregate_candles(candles, target_interval_minutes, source_interval_minutes):
    def aggregate_candles(candles, to_tf: Timeframe):
        # Получаем имя текущей функции
        current_function = inspect.currentframe().f_code.co_name

        aggregated = []
        group = defaultdict(list)
        target_interval_minutes = to_tf.minutes

        interval_ms = (
            candles[1]["ts"] - candles[0]["ts"]
        )  # длительность одной входной свечи в миллисекундах
        input_interval_minutes = interval_ms / 60000  # переводим в минуты

        expected_count = (
            target_interval_minutes / input_interval_minutes
        )  # сколько входных свечей нужно для 1 выходной

        if not expected_count.is_integer():
            print(f"Метод {current_function}: Входные параметры не совместимы")
            return

        expected_count = int(expected_count)

        # Шаг группировки (в миллисекундах)
        interval_ms = target_interval_minutes * 60 * 1000

        for candle in candles:
            # Находим начало нового интервала
            if (to_tf==Timeframe._1week):
                        # Выравниваем по понедельнику 00:00 UTC
                monday_shift = 4 * 24 * 60 * 60 * 1000  # 4 дня в миллисекундах
                week_ms = 7 * 24 * 60 * 60 * 1000       # 1 неделя в миллисекундах

                shifted = candle["ts"] - monday_shift
                start_ts = shifted - (shifted % week_ms) + monday_shift
            else:
                start_ts = candle["ts"] - (candle["ts"] % interval_ms)

            group[start_ts].append(candle)

        for start_ts in sorted(group.keys()):
            group_candles = group[start_ts]
            if not group_candles or (len(group[start_ts]) < expected_count):
                continue

            open_price = group_candles[0]["o"]
            high_price = max(c["h"] for c in group_candles)
            low_price = min(c["l"] for c in group_candles)
            close_price = group_candles[-1]["c"]
            total_vol = sum(c["vol"] for c in group_candles if c["vol"] is not None)
            total_vol_ccy = sum(
                c["volCcy"] for c in group_candles if c["volCcy"] is not None
            )
            total_vol_quote = sum(
                c["volCcyQuote"] for c in group_candles if c["volCcyQuote"] is not None
            )
            aggregated.append(
                {
                    "ts": start_ts,
                    "o": open_price,
                    "h": high_price,
                    "l": low_price,
                    "c": close_price,
                    "vol": total_vol,
                    "volCcy": total_vol_ccy,
                    "volCcyQuote": total_vol_quote,
                    "confirm": 1,
                    "timeFrame": to_tf.label,
                    "quoteCoin": SettingsCoins.quote_coin(),
                }
            )

        return aggregated

    def aggregate_group(group, group_start_ts):
        """
        Превращает список 1m свечей в одну свечу с ts = group_start_ts.
        """
        o = group[0]["o"]
        c = group[-1]["c"]
        h = max(candle["h"] for candle in group)
        l = min(candle["l"] for candle in group)
        vol = sum(candle["vol"] for candle in group)
        volCcy = sum(candle["volCcy"] for candle in group)
        volCcyQuote = sum(candle["volCcyQuote"] for candle in group)
        confirm = 1  # можно поставить 1, т.к. свеча собрана из готовых
        timeFrame = f"{len(group)}m"
        baseCoin = group[0]["baseCoin"]

        return {
            "ts": group_start_ts,
            "o": o,
            "h": h,
            "l": l,
            "c": c,
            "vol": vol,
            "volCcy": volCcy,
            "volCcyQuote": volCcyQuote,
            "confirm": confirm,
            "timeFrame": timeFrame,
            "quoteCoin": SettingsCoins.quote_coin(),
            # Можно оставить остальные индикаторы пустыми (None)
            "ma50": None,
            "ma200": None,
            "ema12": None,
            "ema26": None,
            "macd": None,
            "macd_signal": None,
            "macd_histogram": None,
            "rsi14": None,
            "stochastic_k": None,
            "stochastic_d": None,
        }

    def insert_many_indikator(self, data, name_table: str, return_result: bool = False):
        output_data = 0
        if not data:
            logging.info(ContextLogger.string_context() + " ⚠️ Нет данных для вставки")
            if return_result:
                return output_data

        if not self.connection or not self.connection.is_connected():
            print("⚠️ Нет соединения с базой данных")
            if return_result:
                output_data += 1
                return output_data
            return

        query = f"""
        INSERT INTO {name_table}
        (ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm, timeFrame, quoteCoin, 
         ma50, ma200, ema12, ema26, macd, macd_signal, macd_histogram, rsi14, stochastic_k, stochastic_d)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
        ma50=VALUES(ma50), ma200=VALUES(ma200), ema12=VALUES(ema12), ema26=VALUES(ema26), macd=VALUES(macd), 
        macd_signal=VALUES(macd_signal), macd_histogram=VALUES(macd_histogram), rsi14=VALUES(rsi14),
        stochastic_k=VALUES(stochastic_k), stochastic_d=VALUES(stochastic_d)
        """
        # print(query)
        values = []
        for temp in data:
            values.append(
                (
                    temp["ts"],
                    temp["o"],
                    temp["h"],
                    temp["l"],
                    temp["c"],
                    temp["vol"],
                    temp["volCcy"],
                    temp["volCcyQuote"],
                    temp["confirm"],
                    temp["timeFrame"],
                    temp["quoteCoin"],
                    temp.get("ma50"),
                    temp.get("ma200"),
                    temp.get("ema12"),
                    temp.get("ema26"),
                    temp.get("macd"),
                    temp.get("macd_signal"),
                    temp.get("macd_histogram"),
                    temp.get("rsi14"),
                    temp.get("stochastic_k"),
                    temp.get("stochastic_d"),
                )
            )

        try:
            cursor = self.connection.cursor()
            cursor.executemany(query, values)
            self.connection.commit()
            # print(f"✅ Вставлено записей: {len(values)}")
        except Error as e:
            output_data += 1
            logging.error(ContextLogger.string_context() + f"❌ Ошибка при пакетной вставке: {e}")
            if return_result:
                return output_data
