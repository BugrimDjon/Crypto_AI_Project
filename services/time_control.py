# -*- coding: utf-8 -*-
import time

class TimControl:
    _old_time = None

    @staticmethod
    def frequency_limitation(frequency):
        if TimControl._old_time is None:
            TimControl._old_time = time.time()
            return

        now_time = time.time()
        delta_time = now_time - TimControl._old_time
        wait_time = 1 / frequency

        if delta_time < wait_time:
            time.sleep(wait_time - delta_time)

        TimControl._old_time = time.time()
