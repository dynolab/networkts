import pandas as pd

from networkts.utils.create_features import create_features
from networkts.utils.convert_time import time


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
):
    delta_time = neighbors.index.values[1]-neighbors.index.values[0]
    inds = [time(_*delta_time) for _ in range(neighbors.shape[0]+window_size)]
    dummy_vars = create_features(inds).iloc[window_size:, :]
    dummy_vars.index = neighbors.index
    dummy_vars = pd.concat([neighbors,
                            dummy_vars],
                            axis=1)
    return dummy_vars