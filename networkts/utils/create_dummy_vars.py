import pandas as pd


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