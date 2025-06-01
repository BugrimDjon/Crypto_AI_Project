# from ssl import OP_ENABLE_MIDDLEBOX_COMPAT
# from db import Database
# from services import okx_candles









# db = Database()
# db.connect()

# # ������� ������ (������, �������� ������ ������ �� API)
# db.insert_candle(
#     ts=1234567890, o=1.0, h=1.1, l=0.9, c=1.05,
#     vol=1000, volCcy=1050, volCcyQuote=1100, confirm=1,
#     timeFrame='1m', baseCoin='USDT'
# )

# # ��������� ��������� 10 �������
# candles = db.fetch_candles()
# for c in candles:
#     print(c)

# db.close()

