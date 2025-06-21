class Coins(Enum):
    USDT='USDT'
    EUR='EUR'
    FET='FET'
    BTC='BTC'
    ETH='ETH'
    SOL='SOL'


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




# 🚀 Google Colab шаблон для обучения модели на криптоданных

# ✅ Шаг 1: Подключение Google Диска
from google.colab import drive
drive.mount('/content/drive')

# ✅ Шаг 2: Установка зависимостей
!pip install pandas numpy scikit-learn matplotlib tensorflow==2.13

# ✅ Шаг 3: Импорт библиотек
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import joblib
import os


def run_experiment( 
        table_name: Coins,
        time_frame: Timeframe,
        limit: int = 1000000,
        window_size: int = 60,
        horizon: int = 1,
        model_path=None,
        scaler_path=None,
        return_predictions=False,
        epochs: int = 50,
        learning_rate: float = 0.001,  # регулируем
        dropout: float = 0.2,  # регулируем
        neyro: int = 64,
        df_ready=None,
        offset=None,
        batch_size=64
    ):

    model_name = f"{timeframe.name}_ws{window_size}_hz{horizon}_le_ra{learning_rate}_dr{dropout}_ney{neyro}_offset{offset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path = f"models/{model_name}.h5"
    scaler_path = f"scalers/{model_name}_scalers.pkl"
    df["target"] = df["c"].shift(-horizon)
    df.dropna(inplace=True)

    features = df.drop(columns=["ts", "target"])

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    features_scaled = feature_scaler.fit_transform(features)
    target_scaled = target_scaler.fit_transform(df[["target"]])

    def create_sequences(X, y, window_size, horizon):
        Xs, ys = [], []
        for i in range(len(X) - window_size - horizon + 1):
            Xs.append(X[i : (i + window_size)])
            ys.append(y[i + window_size + horizon - 1])
        return np.array(Xs), np.array(ys)

    X_lstm, y_lstm = create_sequences(
        features_scaled, target_scaled, window_size, horizon
    )

    split_idx = int(0.8 * len(X_lstm))
    X_train, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
    y_train, y_test = y_lstm[:split_idx], y_lstm[split_idx:]

    # ---- Добавлено: перемешивание обучающей выборки ----
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]
    # ------------------------------------------------------

    model = Sequential([
    LSTM(neyro, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2]), implementation=1, unroll=True),
    Dropout(dropout),
    Dense(64, activation="relu"),  # <- новый слой
    Dense(32, activation="relu"),  # <- ещё один
    Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate), loss="mse")

    early_stop = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1,
    )

    # loss = model.evaluate(X_test, y_test)
    # print(f"Final loss (MSE) on test set: {loss}")
    # ✅ Шаг 7: Сохранение модели и скейлера
    model_dir = "/content/drive/MyDrive/Crypto_AI_Project/models"
    scaler_dir = "/content/drive/MyDrive/Crypto_AI_Project/scalers"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)

    mae = mean_absolute_error(y_true.ravel(), y_pred.ravel())

    rmse = root_mean_squared_error(y_true.ravel(), y_pred.ravel())

    df = pd.read_csv(results_file)
    df.loc[len(df)] = [
        datetime.now(), window_size, horizon, epochs,
        loss,val_loss, mae, rmse, model_path, scaler_path, learning_rate, dropout, neyro,offset,batch_size
    ]
    df.to_csv(results_file, index=False)




# ✅ Шаг 4: Загрузка данных (пример: бинарный файл)
data_path = "/content/drive/MyDrive/Crypto_AI_Project/data/df_ready.pkl"
df = pd.read_pickle(data_path)
print("Shape:", df.shape)
df.head()

# ✅ Шаг 5: Подготовка входных данных для LSTM
# Настройки
counter=0
offset=5
for window in [120, 240]: #[30, 45]
    for horizon in [12, 24]:   #[1, 2]
        for learning_rate in [0.001, 0.0005, 0.0001]: #[0.0005, 0.0001]
            for dropout in [0.05, 0.1, 0.15]: #[0.01, 0.05]:
                for neyro in [128, 256, 512]:     #[128, 256]:
                    counter+=1
                    print(f"Проход - {counter}")
                    # tf.debugging.set_log_device_placement(True)
                    if (counter<0):  #613
                        continue
                    if neyro==512:
                        batch_size=32
                    else:
                        batch_size=64
                    run_experiment(
                        table_name=current_coins,
                        timeframe=current_tf,
                        window_size=window,
                        horizon=horizon,
                        epochs=70,
                        learning_rate=learning_rate,
                        dropout=dropout,
                        neyro=neyro,
                        df_ready=df,
                        offset=offset,
                        batch_size=batch_size,
                    )




