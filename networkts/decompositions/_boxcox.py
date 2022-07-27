import numpy as np
from scipy import stats


class BoxcoxTarget(object):
    def __init__(self):
        self.lmbd = None
        self.is_fitted: bool = False
        super().__init__()

    def boxcox_target(self, y):
        y = np.array(y).reshape(-1).tolist()
        self.is_fitted = True
        if (len(y) < 1 or len(y) == y.count(y[0])):
            res, self.lmbd = np.log(y), 0
            return res
        res, self.lmbd = stats.boxcox(y)
        return res

    def inverse_boxcox_target(self, y):
        if not self.is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `target` first."
            )
        if self.lmbd == 0:
            return np.exp(y)
        return np.power(abs(y*self.lmbd+1), 1/self.lmbd)


class NotFittedError(Exception):
    pass