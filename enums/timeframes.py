from enum import Enum

class Timeframe(Enum):
    _1min = ('1m', 1)
    _5min = ('5m', 5)
    _10min = ('10m', 10)
    _15min = ('15m', 15)
    _30min = ('30m', 30)
    _1hour = ('1h', 60)
    _4hour = ('4h', 240)
    _1day = ('1d', 1440)
    _1week = ('1w', 10080)

    @property
    def label(self):
        return self.value[0]

    @property
    def minutes(self):
        return self.value[1]
