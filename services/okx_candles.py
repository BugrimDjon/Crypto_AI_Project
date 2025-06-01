import requests
from enums import coins
from enums import timeframes
from config.SettingsCoins import SettingsCoins
from enums.AfterBefore import AfterBefore

class OkxCandlesFetcher:
    BASE_URL = "https://www.okx.com"
    CANDLES_ENDPOINT = "/api/v5/market/history-candles"

    def __init__(self, instId: coins.Coins, bar: timeframes.Timeframe):
        """
        :param instId: инструмент, например BTC-USDT
        :param bar: таймфрейм, например "1m", "5m", "1h", "1D"
        """
        self.instId = instId
        self.bar = bar

    def fetch_candles(self, limit=100, afterBefore: AfterBefore = None, start_time: str = None):
        url = self.BASE_URL + self.CANDLES_ENDPOINT
        params = {
            "instId": self.instId.value + "-" + SettingsCoins.quote_coin().value,
            "bar": self.bar.value,
            "limit": limit
        }

        if afterBefore is not None and start_time is not None:
            params[afterBefore.value] = start_time
        elif afterBefore is not None or start_time is not None:
            raise ValueError("Оба параметра afterBefore и start_time должны быть заданы одновременно")

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data["code"] != "0":
            raise Exception(f"Ошибка API OKX: {data['msg']}")

        raw_candles = data["data"]
        # print(raw_candles)
        candles = []
        for c in raw_candles:
            candle = {
                "ts": c[0],  # время в ISO8601
                "o": float(c[1]),
                "h": float(c[2]),
                "l": float(c[3]),
                "c": float(c[4]),
                "vol": float(c[5]),
                "volCcy":float(c[6]),
                "volCcyQuote":float(c[7]),
                "confirm":float(c[8])   
            }
            candles.append(candle)

        return candles
