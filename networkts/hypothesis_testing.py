import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from networkts.utils.common import CONF, run_rscript


def accept_iid_based_on_multiple_p_values(p_values: np.ndarray):
    n_rejects = 0
    for i in range(p_values.shape[0]):
        if p_values[i] < 0.05:
            n_rejects += 1
            print(f'Reject at lag {i+1}: p-value {p_values[i]} (implies non-iid)')
    return (n_rejects / p_values.shape[0]) < 0.05


def ljung_box_test(sequence: np.ndarray) -> bool:
    """
    Returns False if null hypothesis (iid process) is rejected at 0.95 conf. level.
    Since we have an array of lags and, thus, a multiple testing problem, we reject more
    than 5% of p_values are smaller than 0.05.
    """
    df = acorr_ljungbox(sequence, lags=10)
    return accept_iid_based_on_multiple_p_values(df['lb_pvalue'].to_numpy())


def mcleod_li_test(sequence: np.ndarray) -> bool:
    """
    Returns False if null hypothesis (iid process) is rejected at 0.95 conf. level.
    Since we have an array of lags and, thus, a multiple testing problem, we reject more
    than 5% of p_values are smaller than 0.05.
    """
    df = run_rscript(CONF['hypothesis_testing']['mcleod_li_test_script_name'],
                     inputs=pd.DataFrame({'values': sequence}))
    return accept_iid_based_on_multiple_p_values(df['p.values'].to_numpy())


def iid_noise_check_based_on_portmanteau_tests(sequence: np.ndarray) -> bool:
    """
    This is only one way to check iid. See more on page 30 in Brockwell, 2016.
    """
    return ljung_box_test(sequence) and mcleod_li_test(sequence)
