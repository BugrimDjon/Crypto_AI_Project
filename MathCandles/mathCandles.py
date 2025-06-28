# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from enums.timeframes import Timeframe

class MathCandles:
    
    def aggregate_with_offset(self, df: pd.DataFrame, tf_minutes: Timeframe, offset_minutes=0):
        """
        Агрегирует минутные свечи в более крупный таймфрейм с учётом смещения.
        """
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)  # <- ВАЖНО: utc=True
        df.set_index("ts", inplace=True)

        offset = pd.Timedelta(minutes=offset_minutes)

        df.sort_index(inplace=True)
        df_agg = df.resample(
            rule=f"{tf_minutes.minutes}T",
            origin="epoch",
            offset=offset
        ).agg({
            "o": "first",
            "h": "max",
            "l": "min",
            "c": "last",
            "vol": "sum",
            "volCcy": "sum",
            "volCcyQuote": "sum"
        })

                # Удалить незавершённые свечи
        max_ts = df.index.max()
        interval = pd.Timedelta(minutes=tf_minutes.minutes)
        df_agg = df_agg[df_agg.index + interval <= max_ts]

        df_agg.dropna(inplace=True)
        df_agg = df_agg.reset_index()
        df_agg["ts"] = df_agg["ts"].astype(np.int64) // 1_000_000  # Возвращаем в миллисекунды

        return df_agg

   
    
    def add_indicators(self, df:pd.DataFrame):
        """ 
        Добавляет стандартные индикаторы к датафрейму свечей.
        """
        df = df.copy()
        df["ma50"] = df["c"].rolling(window=50).mean()
        df["ma200"] = df["c"].rolling(window=200).mean()
        df["ema12"] = df["c"].ewm(span=12, adjust=False).mean()
        df["ema26"] = df["c"].ewm(span=26, adjust=False).mean()

        df["macd"] = df["ema12"] - df["ema26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        # RSI (14)
        delta = df["c"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df["rsi14"] = 100 - (100 / (1 + rs))

        
        df["macd_histogram"] = df["macd"] - df["macd_signal"]


        # Стохастик
        low14 = df["l"].rolling(window=14).min()
        high14 = df["h"].rolling(window=14).max()
        df["stochastic_k"] = 100 * (df["c"] - low14) / (high14 - low14 + 1e-9)
        df["stochastic_d"] = df["stochastic_k"].rolling(window=3).mean()

        return df


    def generate_multi_shift_features(self, df_1min: pd.DataFrame, tf_minutes: Timeframe, offsets: list[int]):
        all_dfs = []
        for offset in offsets:
            agg = self.aggregate_with_offset(df_1min, tf_minutes=tf_minutes, offset_minutes=offset)
            # print(agg["ts"].apply(lambda x: pd.to_datetime(x, unit="ms", utc=True)))

            agg = self.add_indicators(agg)
            agg["offset"] = offset
            all_dfs.append(agg)

        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df.dropna(inplace=True)
        return merged_df

