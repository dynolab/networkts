import pandas as pd
import numpy as np
from datetime import timedelta

from networkts.utils.create_features import create_features


def create_dummy_vars(
    index,
    period
):
    cols = []
    for i in range(period):
        cols.append(f'dv_{i}')
    
    dummy_vars = pd.DataFrame(0, index=range(len(index)), columns=cols)

    for el in dummy_vars.index:
        dummy_vars[f'dv_{(el)%period}'].iloc[el] = 1
    
    dummy_vars.set_index(index, inplace=True)
    return dummy_vars

def create_xgb_dummy_vars(
    neighbors: pd.DataFrame,
    window_size: int,
    delta_time: int,
):
    inds = [neighbors.index[-1] + timedelta(minutes=i*delta_time) for i in range(1, window_size+1)]
    inds = np.concatenate([neighbors.index, pd.to_datetime(inds)])
    dummy_vars = create_features(inds).iloc[window_size:, :]
    dummy_vars.index = neighbors.index
    dummy_vars = pd.concat([neighbors,
                            dummy_vars],
                            axis=1)
    return dummy_vars