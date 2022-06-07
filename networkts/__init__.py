import os
import logging

import yaml
import matplotlib.pyplot as plt

from .utils import common

#  Load config
try:
    with open(os.path.join('config', 'config.yaml')) as f:
        common.CONF = yaml.safe_load(f)
except FileNotFoundError as e:
    raise FileNotFoundError('Configuration file config/config.yaml not found. '
                            'Please create one by copying and renaming an '
                            'example: config/config.yaml.example')

#  Configure loggers
logging.basicConfig(format=common.CONF['logger']['format'],
                    datefmt=common.CONF['logger']['date_format'],
                    level=logging.INFO)

#  Configure matplotlib
plt.style.use('config/default.mplstyle')
