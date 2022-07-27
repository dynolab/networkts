import numpy as np


class LogTarget(object):
    def __init__(self):
        super().__init__()

    def log_target(self, y):
        return np.log(y)

    def inverse_log_target(self, y):
        return np.exp(y)
