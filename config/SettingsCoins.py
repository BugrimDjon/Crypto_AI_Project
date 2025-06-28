# -*- coding: utf-8 -*-
from enums import coins

class SettingsCoins:
    
    @staticmethod
    def quote_coin() -> coins.Coins:
        return coins.Coins.USDT.name

    