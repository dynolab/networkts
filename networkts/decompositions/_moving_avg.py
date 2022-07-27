import pandas as pd


class MATarget(object):
    def __init__(
        self,
        num_previous_points: int = 100
    ):
        self.num_previous_points = num_previous_points
        super().__init__()

    def ma_target(self, y):
        data = pd.DataFrame(y)
        data = data.rolling(
                    self.num_previous_points,
                    min_periods=1
                    ).mean()
        return data.values

    def inverse_ma_target(self, y):
        # do nothing
        return y