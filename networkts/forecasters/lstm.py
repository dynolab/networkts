import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM

from networkts.utils.create_dataset import create_dataset

from networkts.base import BaseForecaster, Timeseries, as_numpy_array
from networkts.utils.create_dataset import create_dataset


class NtsLstm(BaseForecaster):
    LOGGER = logging.getLogger(__qualname__)

    def __init__(
                self,
                lstm_units=64,
                drop=0.1,
                look_back=1,
                epochs=100,
                batch_size=32
                ):
        self.lstm_units = lstm_units
        self.drop = drop
        self.look_back = look_back
        self.epochs = epochs
        self.batch_size = batch_size
        super().__init__()

    def _fit(
        self,
        X: Timeseries,
        y: Timeseries,
    ):
        # y needs be more than test_size (n_timesteps) in 2 times

        model = Sequential()
        model.add(LSTM(self.lstm_units, input_shape=(1, self.look_back)))
        model.add(Dense(self.look_back))
        model.compile(loss='mse', optimizer='adam', metrics=['mape', 'mae'])

        '''
        model = Sequential()
        model.add(LSTM(
                    units=self.lstm_units,
                    return_sequences=True,
                    input_shape=(1, self.look_back)
                    ))
        model.add(Dropout(rate=self.drop))
        model.add(LSTM(units=self.lstm_units, return_sequences=True))
        model.add(Dropout(rate=self.drop))
        model.add(LSTM(units=self.lstm_units, return_sequences=True))
        model.add(Dropout(rate=self.drop))
        model.add(LSTM(units=self.lstm_units, return_sequences=False))
        model.add(Dropout(rate=self.drop))
        model.add(Dense(self.look_back))
        model.compile(loss='mse', optimizer='adam', metrics=['mape', 'mae'])
        '''

        train_x, train_y = create_dataset(
                                dataset=as_numpy_array(y).reshape(-1, 1),
                                look_back=self.look_back
                                )
        train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])

        model.fit(
                train_x,
                train_y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=False
                )
        self._model = model
        return self

    def _predict(
        self,
        X: Timeseries,
    ):
        n_timesteps = X.shape[0]
        y = as_numpy_array(self._y)
        y = y[-n_timesteps:]
        y = y.reshape(1, 1, n_timesteps)
        y_pred = self._model.predict(y).reshape(-1)
        return y_pred
