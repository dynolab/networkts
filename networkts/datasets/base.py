from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from copy import copy
from typing import Tuple, List, Dict

import numpy as np
import networkx as nx
import pandas as pd


Pair = Tuple[str, str]


class TimeseriesType(Enum):
    NODE = auto()
    NODE_PAIR = auto()
    EDGE = auto()


class CovariateType(Enum):
    AVERAGE = auto()


@dataclass
class NetworkTimeseries:
    data: pd.DataFrame  # each column corresponds to the node/edge/etc. name
    data_label: str = 'Traffic'
    time_label: str = 'Time'


@dataclass
class Dataset:
    name: str
    topology: nx.Graph # node/edge names are the same as in node/edge dataframes
    node_timeseries: NetworkTimeseries = None  # each column corresponds to the node name
    edge_timeseries: NetworkTimeseries = None  # each column corresponds to the edge name
    node_pair_timeseries: NetworkTimeseries = None  # each column corresponds to the name of a pair of nodes
    delta_time: int = None
    period: int = None
    rescale: int = None

    @property
    def node_count(self) -> int:
        return self.topology.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.topology.number_of_edges()

    @property
    def node_names(self) -> List[str]:
        return [n for n, _ in self.topology.nodes.items()]

    @property
    def edge_names(self) -> List[Tuple[str, str]]:
        return list(self.topology.edges)

    def timeseries_by_type(self, ts_type: TimeseriesType) -> pd.DataFrame:
        if ts_type == TimeseriesType.NODE:
            return self.node_timeseries
        elif ts_type == TimeseriesType.NODE_PAIR:
            return self.node_pair_timeseries
        elif ts_type == TimeseriesType.EDGE:
            return self.edge_timeseries
        raise ValueError(f'Unknown timeseries type: {ts_type}')

    @abstractmethod
    def make_edge_name(self,
                       source_node: str,
                       dest_node: str,
                       ) -> str:
        raise NotImplementedError('Must be implemented in a child class')

    @abstractmethod
    def make_node_pair_name(self,
                            source_node: str,
                            dest_node: str,
                            ) -> str:
        raise NotImplementedError('Must be implemented in a child class')

    def timeseries_from_node_pairs(self,
                                   source_node: str = None,
                                   dest_node: str = None,
                                   ) -> pd.DataFrame:
        # Example:
        # d.timeseries_from_node_pairs(dest_node='0')

        assert not (source_node is None and dest_node is None)
        raw_dict = {}

        def _put_pair_into_dict(source_node, dest_node):
            node_pair = self.make_node_pair_name(source_node, dest_node)
            raw_dict[node_pair] = self.node_pair_timeseries.data[node_pair]

        if source_node is not None:
            for dest_node in self.node_names:
                _put_pair_into_dict(source_node, dest_node)
        elif dest_node is not None:
            for source_node in self.node_names:
                _put_pair_into_dict(source_node, dest_node)
        else:
            _put_pair_into_dict(source_node, dest_node)
        return pd.DataFrame(raw_dict)

    def shortest_distances(self) -> Dict[Pair, int]:
        dist = {}
        for i in self.topology.nodes:
            for j in self.topology.nodes:
                if i != j:
                    dist[(i, j)] = len(nx.shortest_path(self.topology,
                                                        source=i,
                                                        target=j,
                                                        method='dijkstra')) - 1
        return dist

    def correlation_coefficients(self,
                                 ts_type: TimeseriesType = TimeseriesType.NODE,
                                 names: List[str] = None,
                                 covariate_for_partial_corr: CovariateType = None,
                                 ) -> np.ndarray:
        # Example:
        # d.correlation_coefficients(ts_type=TimeseriesType.NODE_PAIR, names=['(0,1)', '(1,2)', '(0,2)'])

        def _compute_corr_coef(obj):
            if covariate_for_partial_corr is None:
                return obj.corr()
            else:
                import pingouin as pg
                
                df = pd.DataFrame(obj)
                covar_str = covariate_for_partial_corr
                if covariate_for_partial_corr == CovariateType.AVERAGE:
                    df[covar_str] = obj.mean(axis=1)
                else:
                    raise NotImplementedError('Must be implemented')
                corr_values = np.zeros((len(names), len(names)))
                for i, n_one in enumerate(names):
                    for j, n_two in enumerate(names):
                        corr_values[i, j] = pg.partial_corr(data=df, x=n_one, y=n_two, covar=covar_str)
                return pd.DataFrame(data=corr_values, index=names, columns=names)

        ts = self.timeseries_by_type(ts_type)
        if names is None:
            corr = _compute_corr_coef(ts.data)
        else:
            corr = _compute_corr_coef(ts.data[names])
        #  Take the lower triangular part of the correlation matrix
        #  (excluding the diagonal) and flatten it into an array  
        return corr.values[np.tril_indices(corr.shape[0], k=-1)]

    def correlation_coefficients_by_pairs(self,
                                          ts_type: TimeseriesType = TimeseriesType.NODE,
                                          pairs: List[Pair] = None,
                                          ) -> Dict[Pair, float]:
        ts = self.timeseries_by_type(ts_type)
        res = {}
        for node_pair in pairs:
            res[node_pair] = ts.data[node_pair[0]].corr(ts.data[node_pair[1]])
        return res


def create_output_traffic_dataset(dataset: Dataset) -> Dataset:
    output_dataset = copy(dataset)
    output_dataset.name += ' (output node traffic)' 
    raw_dict = {}
    for n_source in dataset.node_names:
        node_pairs_df = dataset.timeseries_from_node_pairs(source_node=n_source)
        raw_dict[n_source] = node_pairs_df.sum(axis=1)
    output_dataset.node_timeseries = NetworkTimeseries(
        data=pd.DataFrame(raw_dict),
        data_label='Output traffic',
    )
    return output_dataset


def create_input_traffic_dataset(dataset: Dataset) -> Dataset:
    # TODO: too similar to create_output_traffic_dataset
    input_dataset = copy(dataset)
    input_dataset.name += ' (output node traffic)' 
    raw_dict = {}
    for n_dest in dataset.node_names:
        node_pairs_df = dataset.timeseries_from_node_pairs(dest_node=n_dest)
        raw_dict[n_dest] = node_pairs_df.sum(axis=1)
    input_dataset.node_timeseries = NetworkTimeseries(
        data=pd.DataFrame(raw_dict),
        data_label='Input traffic',
    )
    return input_dataset
