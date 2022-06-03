import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from networkts.datasets.base import Dataset, TimeseriesType, CovariateType
from networkts.utils.common import inverse_dict
from networkts.utils.plotting import DistributionSummaryPlotType


def plot_correlation_histogram(dataset: Dataset,
                               ax: matplotlib.axes.Axes,
                               type_: TimeseriesType,
                               **hist_kwargs,
                               ):
    corr_values = dataset.correlation_coefficients(type_)
    ax.hist(corr_values, **hist_kwargs)
    ax.set_xlabel('Correlation coefficients')
    if type_ == TimeseriesType.NODE:
        ax.set_ylabel('Count of pairs')
    elif type_ == TimeseriesType.EDGE:
        ax.set_ylabel('Count of edges')


def plot_node_correlation_distribution_against_distance_between_nodes(dataset: Dataset,
                                                                      ax: matplotlib.axes.Axes,
                                                                      covariate_for_partial_corr: CovariateType = None,
                                                                      dist_plot_type: DistributionSummaryPlotType = DistributionSummaryPlotType.BOXPLOT,
                                                                      **plot_kwargs
                                                                      ):
    #  This function can be used to plot corr distrs as a function of distance
    #  for input/output traffic and any other time series on nodes    
    pair_to_dist_dict = dataset.shortest_distances()
    dist_to_pair_list_dict = inverse_dict(pair_to_dist_dict)
    distances = sorted(dist_to_pair_list_dict.keys())
    correlation_coefficients = []
    for d in distances:
        corr_coef_values = dataset.correlation_coefficients_by_pairs(type_=TimeseriesType.NODE,
                                                                     pairs=dist_to_pair_list_dict[d],
                                                                     covariate_for_partial_corr=covariate_for_partial_corr)
        correlation_coefficients.append(np.array(corr_coef_values.values()))
    if dist_plot_type == DistributionSummaryPlotType.BOXPLOT:
        sns.boxplot(x=distances,
                    y=correlation_coefficients,
                    ax=ax,
                    **plot_kwargs
                    )
    elif dist_plot_type == DistributionSummaryPlotType.SCATTER:
        ax.scatter(distances,
                   correlation_coefficients,
                   **plot_kwargs
                   )
        ax.axhline(y=np.mean(correlation_coefficients),
                   color='r',
                   linestyle='-',
                   label='Average'
                   )
    ax.set_xlabel('Distance')
    if covariate_for_partial_corr is None:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel(f'Partial correlation (covariate is {covariate_for_partial_corr})')
    ax.set_title('Traffic')


def plot_correlation_between_object_and_neighbors(dataset: Dataset,
                                                  ax: matplotlib.axes.Axes,
                                                  type_: TimeseriesType,
                                                  covariate_for_partial_corr: CovariateType = None,
                                                  dist_plot_type: DistributionSummaryPlotType = DistributionSummaryPlotType.BOXPLOT,
                                                  **plot_kwargs
                                                  ):
    correlation_coefficients = {}
    
    def _put_corr_coeffs(key, pairs):
        corr_coef_values = dataset.correlation_coefficients_by_pairs(type_=type_,
                                                                     pairs=pairs,
                                                                     covariate_for_partial_corr=covariate_for_partial_corr)
        correlation_coefficients[key] = np.array(corr_coef_values.values())

    if type_ == TimeseriesType.NODE:
        for n in dataset.node_names:
            pairs = [(n, n_neighbor) for n_neighbor in dataset.topology.neighbors(n)]
            _put_corr_coeffs(n, pairs)
    elif type_ == TimeseriesType.EDGE:
        for e in dataset.edge_names:
            dataset.topology.edges(e[0])
            # Add edges originating from the source node
            pairs = [(e, e_neighbor) for e_neighbor in dataset.topology.edges(e[0]) if e != e_neighbor]
            # Add edges originating from the destination node
            pairs += [(e, e_neighbor) for e_neighbor in dataset.topology.edges(e[1]) if e != e_neighbor]
            _put_corr_coeffs(n, pairs)
    if dist_plot_type == DistributionSummaryPlotType.BOXPLOT:
        sns.boxplot(x=correlation_coefficients.keys(),
                    y=correlation_coefficients.values(),
                    ax=ax,
                    **plot_kwargs
                    )
    elif dist_plot_type == DistributionSummaryPlotType.SCATTER:
        ax.scatter(correlation_coefficients.keys(),
                   correlation_coefficients.values(),
                   **plot_kwargs
                   )
        ax.axhline(y=np.mean(correlation_coefficients.values()),
                   color='r',
                   linestyle='-',
                   label='Average'
                   )
    ax.set_xlabel(f'{type_}')
    if covariate_for_partial_corr is None:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel(f'Partial correlation (covariate is {covariate_for_partial_corr})')
    ax.set_title(f'Traffic correlation between {type_} and its neighbors')


# TODO: this function looks too specific to transport network. How can it be generalized?

def plot_box_corr_dist_nodes_traffic_for_pemsd7(dataset,
                                     k: int = 3,
                                     save_mod: bool = False
                                     ):
    x = dataset.route_distances.reshape(-1)/1000
    y = abs(dataset.speeds.corr().values).reshape(-1)
    key = [f'{i}-{i+k}' for i in range(0, int(max(x)), k)]
    key[-1] = f'{int(max(x)-k)}+'
    x2 = []
    for el in x:
        t = int(el//k) if el<int(max(x)-k) else len(key)-1
        x2.append(key[t])
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=x2, y=y)
    plt.xlabel("Distance, km")
    plt.ylabel("Correlation")
    plt.title("Dependence between corr & distance")
    if save_mod:
        plt.savefig(os.path.join(dataset.config['savefig_path'],
                                'corr_dist_nodes_traffic.png'))
    plt.show()
