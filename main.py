
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
    os.system('cls' if os.name == 'nt' else 'clear')

clear_console()
# открываем соединение с бд
db = Database()
db.connect()
# передача открыбой бд в мнтод
run_servis=Servise(db)

# run_servis.first_load_candles(Coins.FET) # готово
# err=run_servis.check_sequence_timeframes(Coins.FET.value,Timeframe._5min,1717171920000,1748729760000,True)


# print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])  # обрезаем до миллисекунд
# len=1
# while len!=0:
# len=run_servis.recalc_timeframe(Coins.FET, Timeframe._1day, Timeframe._1week)

run_servis.calculation_of_indicators(Coins.FET)


# print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])  # обрезаем до миллисекунд
# зкрываем соединение с бд
db.close()


