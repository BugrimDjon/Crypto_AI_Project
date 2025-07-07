import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import matplotlib.pyplot as plt
import gc
import json
# from services.service import Servise

class ExperimentManager:
    def __init__(self, ai_service):
        self.ai_service = ai_service
        self.results_file = "results/experiment_log.csv"
        os.makedirs("results", exist_ok=True)
        os.makedirs("results/models", exist_ok=True)
        os.makedirs("results/scalers", exist_ok=True)
        os.makedirs("results/history", exist_ok=True)
        if not os.path.exists(self.results_file):
            self._init_log()

    def _init_log(self):
        df = pd.DataFrame(columns=[
            "timestamp", "window_size", "horizon", "epochs",
            "loss","val_loss", "mae", "rmse", "model_path", "scaler_path","learning_rate", "dropout", "neyro","offset","batch_size",
             "Корреляция прогноза с реальной будущей ценой", "Корреляция прогноза с текущей (входной) ценой", "selected_features",
             "target_type"
        ])
        df.to_csv(self.results_file, index=False, encoding="utf-8-sig")

    def run_experiment(self, table_name, timeframe, window_size, horizon,l2_reg,
                       epochs=50, learning_rate=0.001, dropout=0.2, neyro=64, df_ready=None,offset=None,
                       batch_size=64, selected_features=None, target_type=0,
                       ):
        model_name = f"{timeframe.name}_ws{window_size}_hz{horizon}_le_ra{learning_rate:.5e}_dr{dropout:.7f}_ney{neyro}_offset{offset}_tg_type{target_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = f"results/models/{model_name}.h5"
        scaler_path = f"results/scalers/{model_name}_scalers.pkl"
        history_path = f"results/history/{model_name}_history.json"
        # print ("перешли в run_experiment, запускаем train_model_experiment")
        # Обучение
        model,history,batch_size, feature_scaler, target_scaler, y_true, y_pred, corr_with_future, corr_with_now = self.ai_service.train_model_experiment(
            table_name=table_name,
            time_frame=timeframe,
            window_size=window_size,
            horizon=horizon,
            l2_reg=l2_reg,
            epochs=epochs,
            model_path=model_path,
            scaler_path=scaler_path,
            return_predictions=True,
            learning_rate=learning_rate,        # регулируем
            dropout=dropout,                 # регулируем
            neyro=neyro,                   # регулируем
            df_ready=df_ready,
            offset=offset,
            batch_size=batch_size,
            target_type = target_type,
        )

        # Метрики
        # mae = mean_absolute_error(y_true, y_pred)
        # print(">>> y_true[0]:", y_true[0], "type:", type(y_true[0]))
        # print(">>> y_pred[0]:", y_pred[0], "type:", type(y_pred[0]))
        # rmse = sklearn_mse(y_true, y_pred, squared=False)

        # y_pred_real, y_true_real= Servise.inverse_target_to_real(y_pred, y_true, now_price, target_type)
        
        # rmse = sklearn_mse(y_true.ravel(), y_pred.ravel(), squared=False)
        mae = mean_absolute_error(y_true.ravel(), y_pred.ravel())

        rmse = np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))


        # Логирование
        self._log_result(window_size, horizon, history.epoch[-1]+1 , model_path, scaler_path,
                         history.history['loss'][-1],history.history['val_loss'][-1], mae, rmse,
                         learning_rate, dropout, neyro,offset,batch_size,corr_with_future, corr_with_now,
                         selected_features, target_type
                         )
        
        # Сохраняем историю обучения в json
        history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_dict, f, ensure_ascii=False, indent=2)

        print(f"✅ Модель сохранена: {model_path}")
        print(f"📊 MAE: {mae:.6f}, RMSE: {rmse:.6f}")
        del model, history, y_true, y_pred
        gc.collect()
        return mae, rmse, corr_with_now, corr_with_future
        


    def _log_result(self, window_size, horizon, epochs, model_path, scaler_path,
                     loss,val_loss, mae, rmse, learning_rate, dropout, neyro,offset,
                     batch_size, corr_with_future, corr_with_now, selected_features=None, target_type=100,
                     ):
        df = pd.read_csv(self.results_file)

            # Преобразуем список в строку для сохранения
        features_str = ", ".join(selected_features) if selected_features else ""
        df.loc[len(df)] = [
            datetime.now(), window_size, horizon, epochs,
            loss,val_loss, mae, rmse, model_path, scaler_path, learning_rate, dropout,
            neyro,offset,batch_size, corr_with_future, corr_with_now, features_str, target_type
        ]
        df.to_csv(self.results_file, index=False, encoding="utf-8-sig")

    def plot_results(self):
        df = pd.read_csv(self.results_file)
        plt.figure(figsize=(10, 6))
        plt.scatter(df['window_size'], df['mae'], c='blue', label='MAE')
        plt.scatter(df['window_size'], df['rmse'], c='red', label='RMSE')
        plt.xlabel("Window Size")
        plt.ylabel("Error")
        plt.legend()
        plt.grid(True)
        plt.title("Ошибки моделей при разных window_size")
        plt.show()
