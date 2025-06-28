# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# print(f"DB_HOST={DB_HOST}")
# print(f"DB_PORT={DB_PORT}")
# print(f"DB_USER={DB_USER}")
# print(f"DB_PASSWORD={'***' if DB_PASSWORD else None}")
# print(f"DB_NAME={DB_NAME}")
