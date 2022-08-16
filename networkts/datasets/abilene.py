from dataclasses import dataclass
import os
import logging

import networkx as nx
import pandas as pd

from networkts.utils.common import CONF
from networkts.datasets.base import Dataset, NetworkTimeseries


@dataclass
class AbileneDataset(Dataset):
    LOGGER = logging.getLogger(__qualname__)

    # TODO: actually, these two fields should not be None
    # but we cannot use fields without default values after
    # fields with default values (declared in Dataset).
    # Though this can be fixed somehow, need to check  
    routing_matrix: pd.DataFrame = None

    @classmethod
    def from_config(cls,
                    root: str,
                    topology_adjlist_file: str,
                    nodes_traffic: str,
                    edges_traffic: str,
                    root_matrix: str,
                    delta_time: int,
                    period: int,
                    name: str,
                    rescale: int,
                    ):
        root = os.path.normpath(root)
        G = nx.read_adjlist(os.path.join(root,
                                         topology_adjlist_file),
                            create_using=nx.DiGraph)
        node_traffic_df = pd.read_csv(os.path.join(root,
                                                  nodes_traffic),
                                index_col=0)
        edge_traffic_df = pd.read_csv(os.path.join(root,
                                                   edges_traffic),
                                index_col=0)
        routing_matrix = pd.read_csv(os.path.join(root,
                                                  root_matrix),
                                     index_col=0)
#        conf = CONF['datasets']['abilene']
#        root = os.path.normpath(conf['root'])
#        G = nx.read_adjlist(os.path.join(root,
#                                         conf['topology_adjlist_file']),
#                            create_using=nx.DiGraph)
#        e2e_traffic_df = pd.read_csv(os.path.join(root,
#                                                  conf['nodes_traffic']),
#                                index_col=0)
#        edge_traffic_df = pd.read_csv(os.path.join(root,
#                                                   conf['edges_traffic']),
#                                index_col=0)
#        routing_matrix = pd.read_csv(os.path.join(root,
#                                                  conf['root_matrix']),
#                                     index_col=0)
        d = cls(
            topology=G,
            node_timeseries=NetworkTimeseries(
                data=node_traffic_df,
                data_label='Node traffic'),
            edge_timeseries=NetworkTimeseries(
                data=edge_traffic_df,
                data_label='Edge traffic'),
            routing_matrix=routing_matrix,
            delta_time=delta_time,
            period=period,
            name=name,
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
