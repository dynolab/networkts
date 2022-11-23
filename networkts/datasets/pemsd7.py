from dataclasses import dataclass
import os
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import networkx as nx

from networkts.utils.common import CONF
from networkts.datasets.base import Dataset, NetworkTimeseries


@dataclass
class Pemsd7Dataset(Dataset):
    LOGGER = logging.getLogger(__qualname__)

    # TODO: actually, these three fields should not be None
    # but we cannot use fields without default values after
    # fields with default values (declared in Dataset).
    # Though this can be fixed somehow, need to check  
    route_distances: np.ndarray = None
    n_stations: int = None

    @classmethod
    def from_config(cls,
                    root: str,
                    topology_adjlist_file: str,
                    route_distances_file: str,
                    speeds_file: str,
                    url: str,
                    delta_time: int,
                    period: int,
                    name: str,
                    rescale: int,
                    ):
        root = os.path.normpath(root)
        G = nx.read_adjlist(os.path.join(
                              root,
                              topology_adjlist_file
                            ),
                            create_using=nx.DiGraph)
        distances = pd.read_csv(os.path.join(
                                  root,
                                  route_distances_file
                                ),
                                header=None).to_numpy()
        speeds_df = pd.read_csv(os.path.join(
                                  root,
                                  speeds_file
                                ),
                                header=None)
        speeds_df.index = [ (datetime.fromisoformat('2012-05-01 00:00:00') + timedelta(minutes=5*i)).strftime("%Y-%m-%d %H:%M:%S")
                            for i in range(speeds_df.shape[0])]
#        conf = CONF['datasets']['pemsd7']
#        root = os.path.normpath(conf['root'])
#        G = nx.read_adjlist(os.path.join(
#                              root,
#                              conf['topology_adjlist_file']
#                            ),
#                            create_using=nx.DiGraph)
#        distances = pd.read_csv(os.path.join(
#                                  root,
#                                  conf['route_distances_file']
#                                ),
#                                header=None).to_numpy()
#        speeds_df = pd.read_csv(os.path.join(
#                                  root,
#                                  conf['speeds_file']
#                                ),
#                                header=None)
        speeds_df.rename(columns=lambda x: str(x), inplace=True)
        d = cls(
            name=name,
            topology=G,
            node_timeseries=NetworkTimeseries(
                data=speeds_df,
                data_label='Road speeds'),
            route_distances=distances,
            n_stations=distances.shape[0],
            delta_time=delta_time,
            period=period,
            rescale=rescale,
        )
        return d

    def make_edge_name(self,
                       source_node: str,
                       dest_node: str,
                       ) -> str:
        return f'({source_node},{dest_node})'

    def make_node_pair_name(self,
                            source_node: str,
                            dest_node: str,
                            ) -> str:
        return f'({source_node},{dest_node})'

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
        return G
