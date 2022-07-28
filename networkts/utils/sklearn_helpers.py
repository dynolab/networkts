from sklearn.base import BaseEstimator, RegressorMixin


class SklearnWrapperForForecaster(RegressorMixin, BaseEstimator):
    def __init__(self, custom_estimator):
        self.custom_estimator = custom_estimator

    def fit(self, X, y, **kwargs):
        self.custom_estimator.fit(y=y, X=X, **kwargs)
        return self

    def predict(self, X, **kwargs):
        return self.custom_estimator.predict(X=X, **kwargs)


def build_target_transformer(
                        transform_class,
                        pipe_or_estimator,
                        func,
                        inverse_func,
                        check_inverse=False,
                        ):
    return transform_class(
                        pipe_or_estimator,
                        func=func,
                        inverse_func=inverse_func,
                        check_inverse=check_inverse,
                        )
