
from enums.coins import Coins
from database.db import Database
from services.okx_candles import OkxCandlesFetcher
from services.time_control import TimControl
from enums.timeframes import Timeframe
from config.SettingsCoins import SettingsCoins
from enums.AfterBefore import AfterBefore

class Servise:
    def __init__(self, db: Database) -> None:
        self.db=db  

    def first_load_candles(self, coin: Coins, length_in_hours=365*24):
       
        TimControl.frequency_limitation(10)
        fetcher = OkxCandlesFetcher(instId=coin, bar=Timeframe._1min)
        data = fetcher.fetch_candles(limit=60)
        for candle in data:
            candle["timeFrame"] = "1m"
            candle["quoteCoin"] = SettingsCoins.quote_coin().value

        self.db.insert_many_candles(data, coin.value)

        for x in range(length_in_hours): 
            TimControl.frequency_limitation(10)
            data = fetcher.fetch_candles(limit=60,afterBefore=AfterBefore.After, start_time=data[-1]["ts"])
            for candle in data:
                candle["timeFrame"] = "1m"
                candle["quoteCoin"] = SettingsCoins.quote_coin().value

            self.db.insert_many_candles(data, coin.value)

            print(f"Обработано {x*100/length_in_hours:.3f}%. ( {x*60} минут из {length_in_hours*60})")



    def recalc_timeframe(self, baseCoin: Coins, from_tf: Timeframe, to_tf: Timeframe):
        interval_minutes = to_tf.minutes / from_tf.minutes
        if not interval_minutes.is_integer():
            print("Входные таймфреймы не совместимы")
            return
        interval_minutes = int(interval_minutes)
        # Получаем максимальный ts для нужного таймфрейма
        last_ts = self.db.get_max_timestamp( baseCoin.value,timeFrame=to_tf.label, quoteCoin=SettingsCoins.quote_coin())

        if last_ts is None:
            # Если данных нет, возьмём минимум по from_tf 
            candles = self.db.fetch_candles_from_ts(baseCoin.value, from_tf.label, 0, SettingsCoins.quote_coin(),10)
        else:
            # Начинаем с ts следующей свечи
            start_ts = last_ts + interval_minutes * 60 * 1000
            candles = self.db.fetch_candles_from_ts(baseCoin.value, from_tf.label,start_ts, SettingsCoins.quote_coin(),10)
        
        # Аггрегируем
        aggregated_candles = Database.aggregate_candles(candles, to_tf)

        
        # Вставляем обратно
        self.db.insert_many_candles(aggregated_candles, name_table=baseCoin.value)

        print(f"Пересчитано {len(aggregated_candles)} свечей с {from_tf} в {to_tf}")
        return len(aggregated_candles)
