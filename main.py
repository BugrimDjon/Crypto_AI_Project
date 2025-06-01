
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
first_load=Servise(db)

# first_load.first_load_candles(Coins.FET) # готово


print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])  # обрезаем до миллисекунд
len=1
while len!=0:
    len=first_load.recalc_timeframe(Coins.FET, Timeframe._1min,Timeframe._5min)

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])  # обрезаем до миллисекунд
# зкрываем соединение с бд
db.close()


