import os
import logging
import random

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg

from sttf.utils.common import CONF


class Pemsd7Dataset:
    LOGGER = logging.getLogger(__qualname__)

    def __init__(
                self,
                topology: nx.DiGraph = None,
                route_distances: np.ndarray = None,
                speeds: pd.DataFrame = None
                ):
        self.topology = topology
        self.route_distances = route_distances
        self.speeds = speeds
        if not (route_distances is None):
            self.n_stations = self.route_distances.shape[0]
        self.config = CONF['datasets']['pemsd7']
        self.adj_matrix = None

    def from_config(self):
        G = nx.read_adjlist(os.path.join(self.config['root'],
                                         self.config['topology_adjlist_file']),
                            create_using=nx.DiGraph)
        distances = pd.read_csv(
                            os.path.join(
                                self.config['root'],
                                self.config['route_distances_file']
                                ),
                            header=None
                            ).to_numpy()
        speeds_array = pd.read_csv(
                            os.path.join(
                                self.config['root'],
                                self.config['speeds_file']
                                ),
                            header=None
                            )
        self.route_distances = distances
        self.speeds = speeds_array
        self.topology = G
        self.n_stations = self.route_distances.shape[0]

    def compute_adjacency_matrix(
                        self,
                        sigma2,
                        epsilon
                        ):
        """
        Computes the adjacency matrix from distances matrix.
        It uses the formula in
        https://github.com/VeritasYin/STGCN_IJCAI-18#data-preprocessing to
        compute an adjacency matrix from the distance matrix.
        The implementation follows that paper.

        Args:
            route_distances: np.ndarray of shape `(num_routes, num_routes)`.
                Entry `i,j` of this array is the distance between roads `i,j`.
            sigma2: Determines the width of the Gaussian kernel applied to
                the square distances matrix.
            epsilon: A threshold specifying if there is an edge between two
                nodes. Specifically, `A[i,j]=1`
                if `np.exp(-w2[i,j] / sigma2) >= epsilon` and `A[i,j]=0`
                otherwise, where `A` is the adjacency matrix and
                `w2=route_distances * route_distances`

        Returns:
            A boolean graph adjacency matrix.
        """
        route_distances = self.route_distances / 10000.0
        w2 = route_distances**2
        el1 = np.ones((self.n_stations, self.n_stations))
        el2 = np.identity(self.n_stations)
        w_mask = el1 - el2
        self.adj_matrix = (np.exp(-w2 / sigma2) >= epsilon) * w_mask
        return self.adj_matrix

    def build_graph(
                self,
                sigma2=0.1,
                epsilon=0.5
                ):
        A = self.compute_adjacency_matrix(sigma2=sigma2, epsilon=epsilon)
        G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
        self.topology = G
