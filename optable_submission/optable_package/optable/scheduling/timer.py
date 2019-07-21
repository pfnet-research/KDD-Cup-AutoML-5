import time
import psutil
import os

import timeout_decorator


class Timer(object):
    def __init__(self, time_budget, time_remain):
        self.start_time = time.time()
        self.time_budget = time_budget

        self.memorized_time = time.time()
        self.memorized_time_remain = time_remain

        pid = os.getpid()
        self.this_process = psutil.Process(pid)

    def update_time_remain(self, time_remain):
        self.memorized_time = time.time()
        self.memorized_time_remain = time_remain

    def ratio_timeout(self, ratio=0.5):
        return timeout_decorator.timeout(
            ratio * self.time_budget - (time.time() - self.start_time)
        )

    def ratio_remain_time(self, ratio=0.5):
        return ratio * self.time_budget - (time.time() - self.start_time)

    @property
    def time_remain(self):
        time_remain = \
            self.memorized_time_remain \
            - (time.time() - self.memorized_time)
        return time_remain

    @property
    def memory_usage(self):
        memory_usage = self.this_process.memory_info().rss/(2.**30)
        return memory_usage

    def print(self, msg):
        time_remain = \
            self.memorized_time_remain \
            - (time.time() - self.memorized_time)
        time_spended = self.time_budget - time_remain
        print(f"INFO  "
              f"[time_remain: {time_remain} time_spended: {time_spended}] "
              f"{msg}")

    def print_memory_usage(self):
        memory_usage = self.this_process.memory_info().rss/(2.**30)
        self.print("uss: {} GB".format(self.this_process.memory_full_info().uss / (2.**30)))
        msg = "memory usage: {} GB".format(memory_usage)
        self.print(msg)
