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

class AIModelService:
    def __init__(self, db: Database) -> None:
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.db = db



