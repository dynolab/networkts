import os
import logging

import yaml
import matplotlib.pyplot as plt

from .utils import common

#  Load config
with open(os.path.join('config', 'config.yaml')) as f:
    common.CONF = yaml.safe_load(f)

#  Configure loggers
logging.basicConfig(format=common.CONF['logger']['format'],
                    datefmt=common.CONF['logger']['date_format'],
                    level=logging.INFO)

#  Configure matplotlib
plt.style.use('config/default.mplstyle')
