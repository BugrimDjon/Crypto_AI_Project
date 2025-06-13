# -*- coding: utf-8 -*-
# Уменьшение потребления CPU потоков
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
import threading
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model, Sequential
from Reports.reports import Reports
from datetime import datetime, timedelta

import time
from datetime import datetime
import os
import requests
from enums.AfterBefore import AfterBefore
from enums.coins import Coins
from enums.timeframes import Timeframe
from ssl import OP_ENABLE_MIDDLEBOX_COMPAT
from database.db import Database
from services import okx_candles
from config.SettingsCoins import SettingsCoins
from services.service import Servise
from AI.AIModelService import AIModelService
from datetime import datetime
import tzlocal



def clear_console():
    os.system("cls" if os.name == "nt" else "clear")

clear_console()
# открываем соединение с бд
db = Database()
db.connect()
# передача открыбой бд в мнтод
run_servis = Servise(db)







# **************************НАЧАЛО ГОТОВО

def agregate_table(table_name: Coins):
    print("********************* Расчет 5 минутных таймфреймов************")
    len = run_servis.recalc_timeframe(table_name, Timeframe._1min, Timeframe._5min)
    print("********************* Расчет 10 минутных таймфреймов************")
    len = run_servis.recalc_timeframe(table_name, Timeframe._5min, Timeframe._10min)
    print("********************* Расчет 15 минутных таймфреймов************")
    len = run_servis.recalc_timeframe(table_name, Timeframe._5min, Timeframe._15min)
    print("********************* Расчет 30 минутных таймфреймов************")
    len = run_servis.recalc_timeframe(table_name, Timeframe._15min, Timeframe._30min)
    print("********************* Расчет 1 часовых таймфреймов************")
    len = run_servis.recalc_timeframe(table_name, Timeframe._30min, Timeframe._1hour)
    print("********************* Расчет 4 часовых таймфреймов************")
    len = run_servis.recalc_timeframe(table_name, Timeframe._1hour, Timeframe._4hour)
    print("********************* Расчет  1 дневных  таймфреймов************")
    len = run_servis.recalc_timeframe(table_name, Timeframe._4hour, Timeframe._1day)
    print("********************* Расчет недельных таймфреймов************")
    len = run_servis.recalc_timeframe(table_name, Timeframe._1day, Timeframe._1week)




while 1:
    menu = f"""
            *******************
            Текущая дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            Введите команду:
            1 - Узнать прогноз
            2 - Обучить модель
            3 - Обновить таблицу
            4 - Начать эксперимент по обучению с разными характеристиками
            8 - Показать колличество записей по таймфреймам в БД
            9 - Начать вычитку новой базы данных, агрегацию и расчет индикаторов
            0 - Выход
            *******************
"""

    

    try:
        button = int(input(menu).strip())
    except ValueError:
        print("⚠️ Пожалуйста, введите число.")
        continue

    # button =4
    match button:
        case 1:
            # run_servis.load_model_and_scalers()
            # price_pred = run_servis.predict_price(Coins.FET, Timeframe._30min)
            # print(f"📈 Прогноз цены закрытия: {price_pred:.6f}")
            # for i in range(100):
            # i=1
            # star_time=1749241800000+i*30*1000*60    
            # run_servis.make_forecast(str(star_time))
            run_servis.make_forecast_on_working_models()

        case 2:
            confirm = input("⚠️ Подтвердите обучение модели (y/n): ").strip().lower()
            if confirm == "y":
                model, fs, ts = run_servis.train_model(
                    Coins.FET, Timeframe._30min, window_size=100
                )
                print("✅ Модель успешно обучена и сохранена.")
            else:
                print("❌ Обучение отменено.")
            # model, feature_scaler, target_scaler = run_servis.for_ai(Coins.FET, Timeframe._30min)
        case 3:
            print("🔄 Обновление таблицы...")
            time_for_update = run_servis.data_for_update(Coins.FET)
            err = run_servis.check_sequence_timeframes(
                Coins.FET,
                Timeframe._1min,
                int(time_for_update["time_in_database"]),
                int(time_for_update["current_time_on_the_exchange"]),
                True,
            )
            agregate_table(Coins.FET)
            run_servis.calculation_of_indicators(
                Coins.FET
            )  # расчет индикаторов по всем таймфреймам
        case 4:
            # run_servis.ai_expirement_predictions()
            run_servis.ai_expirement()
        case 8:
            run_servis.repord_db(Coins.FET)
        case 9:
            run_servis.first_load_candles(Coins.FET, (365 * 1.5 * 24))  # готово
            err = run_servis.check_sequence_timeframes(
                Coins.FET, Timeframe._1min, 0, 1748943960000, True
            )
            agregate_table(Coins.FET)
            run_servis.calculation_of_indicators(
                Coins.FET
            )  # расчет индикаторов по всем таймфреймам
            print("✅ Данные загружены, агрегированы, индикаторы рассчитаны.")

        case 0:
            print("👋 Выход из программы.")
            break

        case 11:
            import traceback

            def background_task():
                global ai_training_in_progress
                if ai_training_in_progress:
                    print("⏸️ Обучение уже идёт, новый процесс не запущен.")
                    return
                try:
                    ai_training_in_progress = True
                    print("🧠 Фоновое обучение моделей запущено...")
                    run_servis.ai_expirement()
                    print("✅ Фоновое обучение завершено.")
                except Exception as e:
                    print(f"❌ Ошибка в фоне обучения: {e}")
                    traceback.print_exc()
                finally:
                    ai_training_in_progress = False


            def wait_until_next_interval(interval_minutes=5):
                now = datetime.now()
                next_minute = (now.minute // interval_minutes + 1) * interval_minutes
                next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=next_minute)
                wait_seconds = (next_time - now).total_seconds()
                time.sleep(wait_seconds)


            ai_training_in_progress = False

            try:
                while True:
                    print(f"\n🔁 Запуск очередного цикла в: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                    try:
                        print("🔄 Обновление данных...")
                        time_for_update = run_servis.data_for_update(Coins.FET)
                        err = run_servis.check_sequence_timeframes(
                            Coins.FET,
                            Timeframe._1min,
                            int(time_for_update["time_in_database"]),
                            int(time_for_update["current_time_on_the_exchange"]),
                            True,
                        )
                        print("✅ Обновление завершено.")

                        print("📊 Агрегация таблиц...")
                        agregate_table(Coins.FET)
                        print("✅ Агрегация завершена.")

                        print("📈 Расчет индикаторов...")
                        run_servis.calculation_of_indicators(Coins.FET)
                        print("✅ Индикаторы рассчитаны.")

                        print("🤖 Прогнозирование моделей...")
                        run_servis.make_forecast_on_working_models()
                        print("✅ Прогноз завершен.")

                        print("🚀 Запуск фонового обучения моделей (клавиша 4)...")
                        thread = threading.Thread(target=background_task, daemon=True)
                        thread.start()

                    except Exception as e:
                        print(f"❌ Ошибка в цикле: {e}")
                        traceback.print_exc()

                    print("⏳ Ожидание следующего интервала...")
                    wait_until_next_interval(5)

            except KeyboardInterrupt:
                print("🛑 Завершено пользователем.")
            except Exception as e:
                print(f"❌ Необработанная ошибка в главном цикле: {e}")
                traceback.print_exc()



        case _:
            print("❌ Неизвестная команда. Попробуйте снова.")
            break



# **************************КОНЕЦ ГОТОВО

# зкрываем соединение с бд
db.close()



#  run_servis.first_load_candles(Coins.FET) # готово
# err=run_servis.check_sequence_timeframes(Coins.FET.value,Timeframe._5min,1717171920000,1748729760000,True)


# print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])  # обрезаем до миллисекунд
# len=1
# while len!=0:
# len=run_servis.recalc_timeframe(Coins.FET, Timeframe._1day, Timeframe._1week)

# run_servis.calculation_of_indicators(Coins.FET) #расчет индикаторов по всем таймфреймам

# *********************************************************************************************
# *********************************************************************************************
# *********************************************************************************************

# time_for_update = run_servis.data_for_update(Coins.FET)

# print(
#     "********************* Обновление минутных таймфреймов до текущего времени************")
# err = run_servis.check_sequence_timeframes(
#     Coins.FET.value, Timeframe._1min,
#     int(time_for_update["time_in_database"]),
#     int(time_for_update["current_time_on_the_exchange"]),
#     True
# )


# print(
#     "********************* Расчет индикаторов************")
# run_servis.calculation_of_indicators(Coins.FET) #расчет индикаторов по всем таймфреймам
# # *****************************************************************************************
# # *****************************************************************************************
# # *****************************************************************************************


# # model, feature_scaler, target_scaler = run_servis.for_ai(Coins.FET, Timeframe._1hour)

# run_servis.load_model_and_scalers()
# result = run_servis.out_ai(Coins.FET, Timeframe._1hour)
# print("Прогноз цены закрытия:", result)


# print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])  # обрезаем до миллисекунд

