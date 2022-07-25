import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from networkts.datasets.abilene import AbileneDataset
from networkts.datasets.pemsd7 import Pemsd7Dataset
from networkts.datasets.totem import TotemDataset
from networkts.plots.misc import plot_topology


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    dataset = TotemDataset().from_config()

    fig, ax = plt.subplots()
    plot_topology(ax, dataset.topology, nx.kamada_kawai_layout)
    plt.show()

    dataset = Pemsd7Dataset().from_config()

    fig, ax = plt.subplots()
    plot_topology(ax, dataset.topology, nx.kamada_kawai_layout)
    plt.show()

    dataset = AbileneDataset().from_config()
    
    fig, ax = plt.subplots()
    plot_topology(ax, dataset.topology, nx.kamada_kawai_layout)
    plt.show()
