import os
import warnings
import logging

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate, call

import networkts
from networkts.plots.misc import plot_topology


LOGGER = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='../config', config_name='config')
def main(cfg: DictConfig) -> None:
    LOGGER.info(f'{os.getcwd()}')

    dataset = call(cfg.dataset)    
    fig, ax = plt.subplots()
    plot_topology(ax, dataset.topology, nx.kamada_kawai_layout)
    plt.show()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
