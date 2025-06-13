# -*- coding: utf-8 -*-
import calendar
from enums.timeframes import Timeframe
from datetime import datetime, timedelta
from database.db import Database
from tzlocal import get_localzone
import pandas as pd
import os
from enums.coins import Coins
from database.db import Database
from enums.timeframes import Timeframe
from datetime import timezone
from visual.visual import Visual

class Reports:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.local_tz = get_localzone()
        self.path = "./out/"  # добавим слэш в конец пути

        # Создаем папку, если не существует
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def update_praice_column(self, out_df: pd.DataFrame, table_name, timeframe: Timeframe):
        # Убедимся, что колонка "data" — datetime без tz
        out_df["data"] = pd.to_datetime(out_df["data"])

        # Находим последнее непустое значение в "praice"
        valid_rows = out_df[out_df["praice"].notna()]
        if not valid_rows.empty:
            last_valid_time = valid_rows["data"].max()
        else:
            last_valid_time = out_df["data"].min()

        # Приводим время к UTC без таймзоны
        last_valid_time_utc = (
            last_valid_time.tz_localize(self.local_tz)
            .tz_convert("UTC")
            .tz_localize(None)
        )

        # Текущее время в UTC, округлённое до 5 минут
        now_utc = datetime.utcnow().replace(second=0, microsecond=0)
        now_utc = pd.to_datetime(now_utc).round("5min")

        # Считаем количество 5-минутных интервалов
        diff_minutes = int((now_utc - last_valid_time_utc).total_seconds() // 60)
        limit = diff_minutes // 5

        if limit <= 0:
            print("Новых данных нет.")
            return out_df

        # Запрос в БД
        query = f"""
            SELECT ts, c FROM {table_name.value}
            WHERE timeFrame = %s
            ORDER BY ts DESC
            LIMIT %s;
        """
        params = (timeframe.label, limit)
        result = self.db.query_to_bd(query, params)

        if not result:
            print("Данные из БД не получены.")
            return out_df

        # Создаём DataFrame из результата
        df_prices = pd.DataFrame(result, columns=["ts", "c"])
        df_prices["data"] = (
            pd.to_datetime(df_prices["ts"], unit="ms", utc=True)
            .dt.tz_convert(self.local_tz)
            .dt.tz_localize(None)
        )
        df_prices["data"] = df_prices["data"].dt.round("5min")
        # добавляем 5 мин для визупльного правильного отображения графика
        # так как свеча называется по времени открытия, а на график мы выводим 
        # цену закрытия, которая в данном случае на 5 мин отличается
        df_prices["data"] += pd.Timedelta(minutes=5)

        # Подготовим основной df
        out_df["data"] = pd.to_datetime(out_df["data"]).dt.round("5min")

        # Устанавливаем индекс
        out_df.set_index("data", inplace=True)
        df_prices.set_index("data", inplace=True)

        # Обновляем колонку "praice"
        out_df.update(df_prices[["c"]].rename(columns={"c": "praice"}))

        # Возвращаем индекс обратно
        out_df.reset_index(inplace=True)

        print(f"Обновлено значений: {len(df_prices)}")

        return out_df

    def report_forecast(self, model_results):
        # Преобразуем входной список словарей в DataFrame
        model_results_df = pd.DataFrame(model_results)

        # Локализуем временную метку в UTC и переводим в локальную временную зону
        model_results_df["data_forcast"] = (
            model_results_df["data_forcast"]
            .dt.tz_localize("UTC")
            .dt.tz_convert(self.local_tz)
            .dt.tz_localize(None)
        )

        # Определяем минимальные и максимальные даты прогноза
        date_forcast_max = model_results_df["data_forcast"].max()
        date_forcast_min = model_results_df["data_forcast"].min()

        # Получаем год и месяц для диапазона прогноза
        max_year, max_month = date_forcast_max.year, date_forcast_max.month
        min_year, min_month = date_forcast_min.year, date_forcast_min.month

        # Если прогноз охватывает два месяца, формируем имена для обоих файлов
        if (max_year != min_year) or (max_month != min_month):
            filename_old = f"{min_year}_{min_month}.csv"
            filename = f"{max_year}_{max_month}.csv"
        else:
            filename_old = ""
            filename = f"{max_year}_{max_month}.csv"

        # Полный путь к CSV-файлу
        full_path = os.path.join(self.path, filename)

        # Если файл не существует, создаем новый
        if not os.path.exists(full_path):
            # Формируем список колонок: дата, цена и имена моделей
            columns = ["data", "praice"] + model_results_df["model"].unique().tolist()

            # Создаём пустой DataFrame с указанными колонками
            out_df = pd.DataFrame(columns=columns)

            # Определяем первый и последний день месяца
            start_date = datetime(max_year, max_month, 1)
            last_day = calendar.monthrange(max_year, max_month)[1]
            end_date = datetime(max_year, max_month, last_day, 23, 55)

            # Создаём диапазон дат с шагом 5 минут
            date_range = pd.date_range(start=start_date, end=end_date, freq="5T")

            # Записываем диапазон дат в колонку 'data'
            out_df["data"] = date_range

            # Сохраняем созданный файл
            out_df.to_csv(full_path, index=False)
            print(f"Файл '{filename}' создан.")
        else:
            # Загружаем существующий файл
            out_df = pd.read_csv(full_path)

            # Преобразуем колонку 'data' в формат datetime для корректного сравнения
            out_df["data"] = pd.to_datetime(out_df["data"])

            # Проходим по всем строкам с результатами модели
            for i in range(len(model_results_df)):
                row = model_results_df.iloc[i]

                # Проверяем, что значение 'prais' не пустое
                if pd.notna(row["prais"]):
                    # Обновляем соответствующую ячейку в DataFrame
                    out_df.loc[out_df["data"] == row["data_forcast"], row["model"]] = (
                        row["prais"]
                    )

        out_df = self.update_praice_column(out_df, Coins.FET, Timeframe._5min)
                # Если decimal не сработал, то можно так:
        for col in out_df.columns[1:]:  # пропускаем колонку с датой
            out_df[col] = pd.to_numeric(out_df[col].astype(str).str.replace(',', '.'), errors='coerce')
        out_df.to_csv(full_path, index=False, decimal=",")
        # visual = Visual()
        # visual.plot_price_and_forecasts2(out_df)
        # visual.plot_price_and_forecasts(out_df)

