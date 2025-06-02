# -*- coding: utf-8 -*-
import inspect

class ContextLogger:

    @staticmethod
    def string_context():
        # Получаем стек вызовов
        frame = inspect.currentframe().f_back
        method_name = frame.f_code.co_name

        # Получаем объект self (если есть)
        self_obj = frame.f_locals.get('self', None)
        class_name = self_obj.__class__.__name__ if self_obj else 'НеизвестныйКласс'

        return(f"Класс: {class_name}, Метод: {method_name}")
