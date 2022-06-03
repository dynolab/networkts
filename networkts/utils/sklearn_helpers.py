from functools import partial

from sklearn.base import BaseEstimator, RegressorMixin


class SklearnWrapperForForecaster(RegressorMixin, BaseEstimator):
    def __init__(self, custom_estimator):
        self.custom_estimator = custom_estimator

    def fit(self, X, y, **kwargs):
        self.custom_estimator.fit(y, X, **kwargs)
        return self

    def predict(self, X, n_timesteps=None, **kwargs):
        return self.custom_estimator.predict(n_timesteps, X, **kwargs)


def build_target_transformer(
                            transform_class,
                            pipe_or_estimator,
                            func,
                            inverse_func,
                            check_inverse=False,
                            params=None,
                            inverse_params=None,
                            ):
    return transform_class(
                            pipe_or_estimator,
                            func=partial(
                                    func,
                                    params=params,
                                    ),
                            inverse_func=partial(
                                    inverse_func,
                                    params=inverse_params
                                    ),
                            check_inverse=check_inverse,
                            )
