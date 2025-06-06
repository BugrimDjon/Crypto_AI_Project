import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import matplotlib.pyplot as plt

class ExperimentManager:
    def __init__(self, ai_service):
        self.ai_service = ai_service
        self.results_file = "results/experiment_log.csv"
        os.makedirs("models", exist_ok=True)
        os.makedirs("scalers", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        if not os.path.exists(self.results_file):
            self._init_log()

    def _init_log(self):
        df = pd.DataFrame(columns=[
            "timestamp", "window_size", "horizon", "epochs",
            "loss", "mae", "rmse", "model_path", "scaler_path"
        ])
        df.to_csv(self.results_file, index=False)

    def run_experiment(self, table_name, timeframe, window_size, horizon, epochs=20):
        model_name = f"model_ws{window_size}_hz{horizon}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = f"models/{model_name}.h5"
        scaler_path = f"scalers/{model_name}_scalers.pkl"

        # Обучение
        model, feature_scaler, target_scaler, y_true, y_pred = self.ai_service.for_ai(
            table_name=table_name,
            time_frame=timeframe,
            window_size=window_size,
            horizon=horizon,
            epochs=epochs,
            model_path=model_path,
            scaler_path=scaler_path,
            return_predictions=True
        )

        # Метрики
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)

        # Логирование
        self._log_result(window_size, horizon, epochs, model_path, scaler_path, model.history.history['loss'][-1], mae, rmse)

        print(f"✅ Модель сохранена: {model_path}")
        print(f"📊 MAE: {mae:.6f}, RMSE: {rmse:.6f}")
        return mae, rmse

    def _log_result(self, window_size, horizon, epochs, model_path, scaler_path, loss, mae, rmse):
        df = pd.read_csv(self.results_file)
        df.loc[len(df)] = [
            datetime.now(), window_size, horizon, epochs,
            loss, mae, rmse, model_path, scaler_path
        ]
        df.to_csv(self.results_file, index=False)

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
