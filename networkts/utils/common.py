import os
import subprocess
import logging
from typing import Any

import yaml
import pandas as pd
import matplotlib.pyplot as plt


class Config:
    def __init__(self):
        self._conf_as_dict = None
    
    def set_dict(self, d: dict):
        self._conf_as_dict = d
    
    def __getitem__(self, k: Any):
        if self._conf_as_dict is None:
            raise ConfigNotFoundException(f'Config has not been set. '
                                          f'Call networkts.utils.common.set_config() '
                                          f'to provide the configuration file')
        return self._conf_as_dict[k]


CONF = Config()  # Create an empty config. Must call set_config() to fill it 


def set_config(p: str) -> None:
    global CONF
    #  Load config
    try:
        with open(p) as f:
            CONF.set_dict(yaml.safe_load(f))
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Configuration file {p} not found. '
                                f'Please create one by copying and renaming an '
                                f'example: config/config.yaml.example')
    #  Configure loggers
    logging.basicConfig(format=CONF['logger']['format'],
                        datefmt=CONF['logger']['date_format'],
                        level=logging.INFO)


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


class ConfigNotFoundException(Exception):
    pass