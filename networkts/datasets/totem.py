import os
import logging
import random

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from networkts.utils.common import CONF


class TotemDataset:
    LOGGER = logging.getLogger(__qualname__)

    def __init__(self,
                 topology: nx.DiGraph = None,
                 traffic: pd.DataFrame = None,
                 edges_traffic: pd.DataFrame = None):
        self.topology = topology
        self.traffic = traffic
        self.edges_traffic = edges_traffic
        self.config = CONF['datasets']['totem']

    def from_config(self):
        traffic = pd.read_csv(
                    os.path.join(
                        self.config['root'],
                        self.config['traffic']
                        ),
                    index_col=0)
        edges_traffic = pd.read_csv(
                            os.path.join(
                                self.config['root'],
                                self.config['edges_traffic']
                                ),
                            index_col=0)

        G = nx.read_adjlist(
                    os.path.join(
                        self.config['root'],
                        self.config['topology_adjlist_file']
                        ),
                    create_using=nx.DiGraph)
        self.topology = G
        self.traffic = traffic
        self.edges_traffic = edges_traffic
