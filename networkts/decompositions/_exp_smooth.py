from statsmodels.tsa.holtwinters import ExponentialSmoothing


class ESTarget(object):
    def __init__(
        self,
        smoothing_level: float = 0.1
    ):
        self.smoothing_level = smoothing_level
        super().__init__()

    def es_target(self, y):
        s = ExponentialSmoothing(y).fit(
                                    smoothing_level=self.smoothing_level,
                                    optimized=False
                                    )
        return s.fittedvalues

    def inverse_es_target(self, y):
        # do nothing
        return y