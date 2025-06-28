# -*- coding: cp1251 -*-
import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error



import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error

print("Запуск скрипта...")

# Загружаем переменные окружения из .env
load_dotenv()

print("Переменные окружения:")
print(f"DB_HOST={os.getenv('DB_HOST')}")
print(f"DB_PORT={os.getenv('DB_PORT')}")
print(f"DB_USER={os.getenv('DB_USER')}")
print(f"DB_NAME={os.getenv('DB_NAME')}")

def create_connection():
    try:
        print("Пытаемся подключиться к базе...")
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            port=int(os.getenv('DB_PORT', 3306)),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME')
        )
        if connection.is_connected():
            print("Успешное подключение к базе данных")
            return connection
    except Error as e:
        print(f"Ошибка подключения к базе: {e}")
        return None

def check_connection():
    conn = create_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES;")
            tables = cursor.fetchall()
            print("Список таблиц в базе данных:")
            for table in tables:
                print(table[0])
            cursor.close()
        except Exception as e:
            print(f"Ошибка при выполнении запроса: {e}")
        finally:
            conn.close()
    else:
        print("Не удалось создать соединение")

if __name__ == "__main__":
    check_connection()
