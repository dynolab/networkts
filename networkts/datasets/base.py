from dataclasses import dataclass
from enum import Enum, auto
from copy import copy

import numpy as np
import networkx as nx
import pandas as pd
import pingouin as pg


Pair = tuple[str, str]


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
    """
    topology: nx.Graph

    ABILENE
    dataset.nodes_traffic.columns.values
    dataset.nodes_traffic[el].values
    dataset.config['savefig_path']
    dataset.edges_traffic[el].values

    TOTEM
    dataset.traffic.columns.values
    dataset.traffic[el].values
    dataset.config['savefig_path']

    PEMSD7
    dataset.speeds.columns.values
    dataset.speeds[el].values
    dataset.config['savefig_path']
    """
    name: str
    topology: nx.Graph  # node/edge names are the same as in node/edge dataframes
    node_timeseries: NetworkTimeseries = None  # each column corresponds to the node name
    edge_timeseries: NetworkTimeseries = None  # each column corresponds to the edge name
    node_pair_timeseries: NetworkTimeseries = None  # each column corresponds to the name of a pair of nodes

    @property
    def node_count(self) -> int:
        return self.topology.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.topology.number_of_edges()

    @property
    def node_names(self) -> list[str]:
        return [n for n, _ in self.topology.nodes.items()]

    @property
    def edge_names(self) -> list[tuple(str, str)]:
        return self.topology.edges

    def timeseries_by_type(self, type_: TimeseriesType) -> pd.DataFrame:
        if type_ == TimeseriesType.NODE:
            return self.node_timeseries
        elif type_ == TimeseriesType.NODE_PAIR:
            return self.node_pair_timeseries
        elif type_ == TimeseriesType.EDGE:
            return self.edge_timeseries
        raise ValueError(f'Unknown timeseries type: {type_}')

    def node_pair_name(self,
                       source_node: str,
                       dest_node: str,
                       ) -> str:
        return f'{source_node},{dest_node}'

    def timeseries_from_node_pairs(self,
                                   source_node: str = None,
                                   dest_node: str = None,
                                   ) -> pd.DataFrame:
        assert not (source_node is None and dest_node is None)
        raw_dict = {}

        def _put_pair_into_dict(source_node, dest_node):
            node_pair = self.get_node_pair_name(source_node, dest_node)
            raw_dict[node_pair] = self.node_pair_timeseries[node_pair]

        if source_node is not None:
            for dest_node in self.node_names():
                _put_pair_into_dict(source_node, dest_node)
        elif dest_node is not None:
            for source_node in self.node_names():
                _put_pair_into_dict(source_node, dest_node)
        else:
            _put_pair_into_dict(source_node, dest_node)
        return pd.DataFrame(raw_dict)

    def shortest_distances(self) -> dict[Pair, int]:
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
                                 type_: TimeseriesType = TimeseriesType.NODE,
                                 names: list[str] = None,
                                 covariate_for_partial_corr: CovariateType = None,
                                 ) -> np.ndarray:

        def _compute_corr_coef(obj):
            if covariate_for_partial_corr is None:
                return obj.corr()
            else:
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

        ts = self.timeseries_by_type(type_)
        if names is None:
            corr = _compute_corr_coef(ts.data)
        else:
            corr = _compute_corr_coef(ts.data[names])
        #  Take the lower triangular part of the correlation matrix
        #  (excluding the diagonal) and flatten it into an array  
        return corr.values[np.tril_indices(corr.shape[0], k=-1)]

    def correlation_coefficients_by_pairs(self,
                                          type_: TimeseriesType = TimeseriesType.NODE,
                                          pairs: list[Pair] = None,
                                          ) -> dict[Pair, float]:
        ts = self.timeseries_by_type(type_)
        res = {}
        for node_pair in pairs:
            res[node_pair] = ts.data[node_pair[0]].corr(ts.data[node_pair[1]])
        return res


def create_output_traffic_dataset(dataset: Dataset) -> Dataset:
    output_dataset = copy(dataset)
    raw_dict = {}
    for n_source in dataset.node_names():
        node_pairs_df = dataset.get_timeseries_from_node_pairs(source_node=n_source)
        raw_dict[n_source] = node_pairs_df.sum(axis=1)
    output_dataset.node_timeseries = pd.DataFrame(raw_dict)
    return output_dataset


def create_input_traffic_dataset(dataset: Dataset) -> Dataset:
    # TODO: too similar to create_output_traffic_dataset
    output_dataset = copy(dataset)
    raw_dict = {}
    for n_dest in dataset.node_names():
        node_pairs_df = dataset.get_timeseries_from_node_pairs(dest_node=n_dest)
        raw_dict[n_dest] = node_pairs_df.sum(axis=1)
    output_dataset.node_timeseries = pd.DataFrame(raw_dict)
    return output_dataset
