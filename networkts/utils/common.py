import os
import subprocess

import pandas as pd


CONF = None  # Only declare here. Define in __init__.py 


def inverse_dict(d: dict) -> dict:
    #  Here we assume that values may be non-unique and, thus, the inverse mapping is value -> list[keys]
    inv_dict = {}
    for k, v in d.items():
        inv_dict[v] = inv_dict.get(v, []) + [k]
    return inv_dict


def run_rscript(
        name: str,
        inputs: pd.DataFrame = None,
        input_filename='_temp_input.csv',
        output_filename='_temp_output.csv'
    ):
    if inputs is not None:
        inputs.to_csv(input_filename, index=False)
        command_line = f'Rscript.exe {os.path.join(CONF["rscripts_dir"], name)} {input_filename} {output_filename}'
    else:
        command_line = f'Rscript.exe {os.path.join(CONF["rscripts_dir"], name)}'
    res = subprocess.run(command_line, shell=True)
    with open(output_filename, 'r') as f:
        df = pd.read_csv(f)
    for name in (input_filename, output_filename):
        os.remove(name)
    return df
