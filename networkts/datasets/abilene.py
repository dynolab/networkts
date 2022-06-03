import os
import logging
import random

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sttf.utils.common import CONF


class AbileneDataset:
    LOGGER = logging.getLogger(__qualname__)

    def __init__(
            self,
            topology: nx.Graph = None,
            nodes_traffic: pd.DataFrame = None,
            edges_traffic: pd.DataFrame = None
            ):

        self.topology = topology
        self.nodes_traffic = nodes_traffic
        self.edges_traffic = edges_traffic
        self.rout_matrix = None
        self.config = CONF['datasets']['abilene']

    def from_config(self):
        G = nx.read_adjlist(os.path.join(self.config['root'],
                                         self.config['topology_adjlist_file']),
                            create_using=nx.DiGraph)
        n_traffic = pd.read_csv(os.path.join(self.config['root'],
                                             self.config['nodes_traffic']),
                                index_col=0)
        e_traffic = pd.read_csv(os.path.join(self.config['root'],
                                             self.config['edges_traffic']),
                                index_col=0)
        r_matrix = pd.read_csv(os.path.join(self.config['root'],
                                            self.config['root_matrix']),
                               index_col=0)
        self.topology = G
        self.nodes_traffic = n_traffic
        self.edges_traffic = e_traffic
        self.rout_matrix = r_matrix
