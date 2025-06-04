# -*- coding: utf-8 -*-
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


def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


clear_console()
# открываем соединение с бд
db = Database()
db.connect()
# передача открыбой бд в мнтод
run_servis = Servise(db)

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
print(
    "********************* Расчет 5 минутных таймфреймов************")
len=run_servis.recalc_timeframe(Coins.FET, Timeframe._1min, Timeframe._5min)
print(
    "********************* Расчет 10 минутных таймфреймов************")
len=run_servis.recalc_timeframe(Coins.FET, Timeframe._5min, Timeframe._10min)
print(
    "********************* Расчет 15 минутных таймфреймов************")
len=run_servis.recalc_timeframe(Coins.FET, Timeframe._5min, Timeframe._15min)
print(
    "********************* Расчет 30 минутных таймфреймов************")
len=run_servis.recalc_timeframe(Coins.FET, Timeframe._15min, Timeframe._30min)
print(
    "********************* Расчет 1 часовых таймфреймов************")
len=run_servis.recalc_timeframe(Coins.FET, Timeframe._30min, Timeframe._1hour)
print(
    "********************* Расчет 4 часовых таймфреймов************")
len=run_servis.recalc_timeframe(Coins.FET, Timeframe._1hour, Timeframe._4hour)
print(
    "********************* Расчет  1 дневных  таймфреймов************")
len=run_servis.recalc_timeframe(Coins.FET, Timeframe._4hour, Timeframe._1day)
print(
    "********************* Расчет недельных таймфреймов************")
len=run_servis.recalc_timeframe(Coins.FET, Timeframe._1day, Timeframe._1week)


print(
    "********************* Расчет индикаторов************")
run_servis.calculation_of_indicators(Coins.FET) #расчет индикаторов по всем таймфреймам

# print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])  # обрезаем до миллисекунд
# зкрываем соединение с бд
db.close()
