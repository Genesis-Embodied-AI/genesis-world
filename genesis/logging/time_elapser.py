import threading
import time

import numpy as np

from genesis.utils.emoji import get_clock


class TimeElapser:
    """
    A tool that can be called with `with` statement, and starts a separate thread that continuously outputs logger message with elasped time.
    """

    def __init__(self, logger, refresh_rate, end_msg):
        self.logger = logger
        self.dt = 1.0 / refresh_rate
        self.n = np.ceil(np.log10(refresh_rate)).astype(int)
        self._stop = threading.Event()

        self.last_logger_output = self.logger.last_output
        # re-print the logger message without the reset character
        self.start_msg = self.last_logger_output[:-4]
        # adding back the reset character
        self.end_msg = end_msg + self.last_logger_output[-4:] + "\n"

    # XXX: Kept for more debug options.
    # def __enter__(self):
    #     self.thread = threading.Thread(target=self.run)
    #     self.thread.start()
    #     return self

    # def __exit__(self, exc_type, exc_value, traceback):
    #     self._stop.set()
    #     self.thread.join()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def run(self):
        # re-print logger msg from the beginning of the previous line
        self.logger.raw("\x1b[1F" + self.start_msg + " ")
        t_start = time.perf_counter()
        t_elapsed = time.perf_counter() - t_start
        self.logger.raw(f"~<{t_elapsed:.{self.n}f}s>~ {get_clock(t_elapsed)} ")
        prev_width = len(f"{t_elapsed:.{self.n}f}s ") + 3
        while not self._stop.is_set():
            time.sleep(self.dt)
            with self.logger.lock_timer():
                t_elapsed = time.perf_counter() - t_start
                # check if something in the main thread has moved the cursor to a new line
                if self.logger._is_new_line:
                    self.logger.raw(self.start_msg + " ")
                else:
                    self.logger.raw("\b" * prev_width)
                self.logger.raw(f"~<{t_elapsed:.{self.n}f}s>~ {get_clock(t_elapsed)} ")
                prev_width = len(f"{t_elapsed:.{self.n}f}s ") + 3
        self.logger.raw("\b\b\b✅ " + self.end_msg)
