import warnings

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted, _allclose_dense_sparse
from sklearn.utils import check_array, _safe_indexing
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import TransformedTargetRegressor


class TSFunctionTransformer(FunctionTransformer):
    def __init__(
                self,
                func=None,
                inverse_func=None,
                *,
                validate=False,
                accept_sparse=False,
                check_inverse=True,
                kw_args=None,
                inv_kw_args=None,
                ):
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.check_inverse = check_inverse
        self.kw_args = kw_args
        self.inv_kw_args = inv_kw_args

    def _transform(self, X, func=None, kw_args=None):
        if kw_args is None:
            return func(X)
        return func(X, kw_args)

    def inverse_transform(self, X, lmbd=None):
        if self.validate:
            X = check_array(X, accept_sparse=self.accept_sparse)
        if lmbd is None:
            return self._transform(
                                    X,
                                    func=self.inverse_func,
                                    kw_args=self.inv_kw_args,
                                    )
        return self._transform(X, func=self.inverse_func, kw_args=lmbd)

    def _check_inverse_transform(self, X):
        idx_selected = slice(None, None, max(1, X.shape[0] // 100))
        X_round_trip, self.inv_kw_args = self._transform(
                                                        X[idx_selected],
                                                        func=self.func
                                                        )
        X_round_trip = self.inverse_transform(X_round_trip)
        if not _allclose_dense_sparse(X[idx_selected], X_round_trip):
            warnings.warn(
                "The provided functions are not strictly"
                " inverse of each other. If you are sure you"
                " want to proceed regardless, set"
                " 'check_inverse=False'.",
                UserWarning,
            )

    def fit(self, X, y=None):
        X = self._check_input(X, reset=True)
        fl1 = self.check_inverse
        fl2 = not (self.func is None or self.inverse_func is None)
        if fl1 and fl2:
            self._check_inverse_transform(X)
        return self


class TSTransformedTargetRegressor(TransformedTargetRegressor):
    def __init__(
                self,
                regressor=None,
                *,
                transformer=None,
                func=None,
                inverse_func=None,
                check_inverse=False,
                param=None,
                ):
        self.regressor = regressor
        self.transformer = transformer
        self.func = func
        self.inverse_func = inverse_func
        self.check_inverse = check_inverse
        self.param = param

    def _fit_transformer(self, y):
        fl1 = self.transformer is not None
        fl2 = self.func is not None or self.inverse_func is not None
        if fl1 and fl2:
            er = "'transformer' and functions 'func'"
            er2 = "/'inverse_func' cannot both be set."
            raise ValueError(er+er2)
        elif self.transformer is not None:
            self.transformer_ = clone(self.transformer)
        else:
            if self.func is not None and self.inverse_func is None:
                er = "When 'func' is provided, 'inverse_func' "
                er2 = "must also be provided"
                raise ValueError(er+er2)
            self.transformer_ = TSFunctionTransformer(
                func=self.func,
                inverse_func=self.inverse_func,
                validate=False,
                check_inverse=self.check_inverse,
                inv_kw_args=self.param,
            )

        self.transformer_.fit(y)
        if self.check_inverse:
            idx_selected = slice(None, None, max(1, y.shape[0] // 10))
            y_sel = _safe_indexing(y, idx_selected)
            y_sel_t, self.param = self.transformer_.transform(y_sel)
            if not np.allclose(
                        y_sel,
                        self.transformer_.inverse_transform(y_sel_t)
                        ):
                warnings.warn(
                    "The provided functions or transformer are"
                    " not strictly inverse of each other. If"
                    " you are sure you want to proceed regardless"
                    ", set 'check_inverse=False'",
                    UserWarning,
                )

    def fit(self, X, y, **fit_params):
        y = check_array(
            y,
            accept_sparse=False,
            force_all_finite=True,
            ensure_2d=False,
            dtype="numeric",
            allow_nd=True,
        )

        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self._training_dim = y.ndim

        # transformers are designed to modify X which is 2d dimensional, we
        # need to modify y accordingly.
        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y
        self._fit_transformer(y_2d)

        # transform y and convert back to 1d array if needed
        y_trans, self.param = self.transformer_._transform(
                                                    y_2d,
                                                    func=self.func
                                                    )

        if y_trans.ndim == 2 and y_trans.shape[1] == 1:
            y_trans = y_trans.squeeze(axis=1)

        if self.regressor is None:
            from sklearn.linear_model import LinearRegression

            self.regressor_ = LinearRegression()
        else:
            self.regressor_ = clone(self.regressor)

        self.regressor_.fit(X, y_trans, **fit_params)

        if hasattr(self.regressor_, "feature_names_in_"):
            self.feature_names_in_ = self.regressor_.feature_names_in_
        return self

    def predict(self, X, **predict_params):
        check_is_fitted(self)
        pred = self.regressor_.predict(X, **predict_params)
        if pred.ndim == 1:
            pred_trans = self.transformer_.inverse_transform(
                                                pred.reshape(-1, 1),
                                                lmbd=self.param
                                                )
        else:
            pred_trans = self.transformer_.inverse_transform(
                                                pred,
                                                lmbd=self.param
                                                )
        if (
            self._training_dim == 1
            and pred_trans.ndim == 2
            and pred_trans.shape[1] == 1
        ):
            pred_trans = pred_trans.squeeze(axis=1)
        return pred_trans
