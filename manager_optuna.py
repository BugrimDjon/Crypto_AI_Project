import subprocess
import time

counter = 0
max_runs = 9  # 100 / 5

while True:
    counter += 1
    if counter > max_runs:
        print("Достигнуто максимальное число запусков, завершаем.")
        break

    print(f"Запуск main.py, итерация {counter}")
    retcode = subprocess.call(["python", "main.py"])

    if retcode != 0:
        print(f"Ошибка в main.py, код {retcode}, останавливаем.")
        break

    print("Перезапуск main.py после завершения Optuna...")
    time.sleep(2)
