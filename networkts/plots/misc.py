import os
from typing import Callable, Optional

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

from networkts.datasets.base import Dataset, TimeseriesType


def plot_topology(
    ax: matplotlib.axes.Axes,
    topology: nx.Graph,
    layout: Callable,
    # good options are: nx.kamada_kawai_layout for pemsd7 and totem,
    # nx.spring_layout for abilene
    add_label: bool = True,
    draw_options: Optional[dict] = None,
    label_options: Optional[dict] = None,
):
    if draw_options is None:
        draw_options = {
            'node_size': 600,  # 600 for abilene and totem, 70 for pemsd7
            'node_color': '#c75157',
            'width': 0.5,
            'edge_color': '#aaaaaa',
            'arrows': False,
        }
    if label_options is None:
        label_options = {
            'font_size': 12,
            'font_color': 'white',
        }
    pos = layout(topology)
    nx.draw(topology,
            pos,
            ax=ax,
            **draw_options,
            )
    if add_label:
        nx.draw_networkx_labels(topology,
                                pos,
                                {x: int(float(x))
                                    for x in topology.nodes()},
                                ax=ax,
                                **label_options,
                                )


def plot_network_univariate_timeseries(
    dataset: Dataset,
    ax: matplotlib.axes.Axes,
    type_: TimeseriesType,
    name: str,
    **plot_kwargs,
):
    ts = dataset.timeseries_by_type(type_)
    ax.plot(ts.data[name].values, **plot_kwargs)
    ax.set_xlabel(ts.time_label)
    ax.set_ylabel(ts.data_label)
    ax.set_title()


def plot_nodes_dist_hist_for_pemsd7(
    dataset,
    k: int = 3,
    save_mod: bool = False,
):
    x = dataset.route_distances.reshape(-1)/1000
    key = [f'{i}-{i+k}' for i in range(0, int(max(x)), k)]
    key[-1] = f'{int(max(x)-k)}+'
    x2 = []
    for el in x:
        t = int(el//k) if el < int(max(x)-k) else len(key)-1
        x2.append(key[t])
    plt.figure(figsize=(10, 5))
    plt.hist(x2)
    plt.xlabel("Distance, km")
    plt.ylabel("Road's count")
    plt.title("Distance hist")
    if save_mod:
        plt.savefig(os.path.join(dataset.config['savefig_path'],
                    'dist_hist.png'))
    plt.show()
