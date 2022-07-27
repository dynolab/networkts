from dataclasses import dataclass
import os
import logging

import pandas as pd
import networkx as nx

from networkts.utils.common import CONF
from networkts.datasets.base import Dataset, NetworkTimeseries


@dataclass
class TotemDataset(Dataset):
    LOGGER = logging.getLogger(__qualname__)

    @classmethod
    def from_config(cls,
                    root: str,
                    topology_adjlist_file: str,
                    traffic: str,
                    edges_traffic: str,
                    ):
        root = os.path.normpath(root)
        G = nx.read_adjlist(os.path.join(
                              root,
                              topology_adjlist_file
                            ),
                            create_using=nx.DiGraph)
        e2e_traffic_df = pd.read_csv(os.path.join(
                                       root,
                                       traffic
                                     ),
                                     index_col=0)
        edge_traffic_df = pd.read_csv(os.path.join(
                                        root,
                                        edges_traffic
                                      ),
                                      index_col=0)

#        conf = CONF['datasets']['totem']
#        root = os.path.normpath(conf['root'])
#        G = nx.read_adjlist(os.path.join(
#                              root,
#                              conf['topology_adjlist_file']
#                            ),
#                            create_using=nx.DiGraph)
#        e2e_traffic_df = pd.read_csv(os.path.join(
#                                       root,
#                                       conf['traffic']
#                                     ),
#                                     index_col=0)
#        edge_traffic_df = pd.read_csv(os.path.join(
#                                        root,
#                                        conf['edges_traffic']
#                                      ),
#                                      index_col=0)
        d = cls(
            name='Totem',
            topology=G,
            node_pair_timeseries=NetworkTimeseries(
                data=e2e_traffic_df,
                data_label='E2E traffic'),
            edge_timeseries=NetworkTimeseries(
                data=edge_traffic_df,
                data_label='Edge traffic'),
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
        return f'({source_node}, {dest_node})'
