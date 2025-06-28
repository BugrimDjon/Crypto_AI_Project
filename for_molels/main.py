import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from services.service import Servise   # путь к классу, поправь под себя
# from enums.coins import Coins
# from enums.timeframes import Timeframe  # если нужны
from database.db import Database

# Загружаем переменные окружения, если нужно
# или они уже в системе
# os.environ["DB_HOST"] = "..."
# os.environ["DB_USER"] = "..."

if __name__ == "__main__":


    db = Database()
    db.connect()    
    # передача   открыбой бд в мнтод
    run_servis = Servise(db)

    run_servis.ai_expirement()


# зкрываем соединение с бд
    db.close()