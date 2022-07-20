from functools import wraps
from enum import Enum, auto

import matplotlib.pyplot as plt


class DistributionSummaryPlotType(Enum):
    BOXPLOT = auto()
    SCATTER = auto()


def savefig(path_to_save, dpi=200):
    def inner_decorator(plotting_func):
        @wraps(plotting_func)
        def wrapper(*args, **kwargs):
            res = plotting_func(*args, **kwargs)
            plt.savefig(path_to_save, dpi=dpi)
            return res
        return wrapper
    return inner_decorator


def showfig(plotting_func):
    @wraps(plotting_func)
    def wrapper(*args, **kwargs):
        res = plotting_func(*args, **kwargs)
        plt.show()
        return res
    return wrapper
