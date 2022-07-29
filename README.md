# networkts
Analysis and forecasting of time series defined on networks

# Getting started

## Installation

This package can be installed directly from github via
```bash
$ pip install git+https://github.com/dynolab/networkts.git
```
All the requirements must be installed automatically except for prophet which must be installed manually from conda repositories:
```bash
$ conda install -c conda-forge prophet
```
If you want to update the `networkts` package to have the most fresh, please first uninstall it and then install it again.

## Configuration

The package is intended to be used within a hydra application. Check [config examples](/examples/config) and a [hydra application examples](/examples/hydra_applications).

# Datasets

Out of the box, `networkts` provides classes for three popular datasets: Abilene, Totem and PeMSD7 (`networkts.datasets.AbileneDataset`, `networkts.datasets.TotemDataset` and `networkts.datasets.Pemsd7Dataset` respectively). Each class has `from_config()` method which must be called to instantiate the class. It takes the path to a dataset as one its arguments: to download suitably preprocessed datasets, use the following links:

| Dataset | Link |
|---------|------|
| Abilene | https://onebox.huawei.com/p/6230420e276fe6d361b0d48966785736 |
| PeMSD7 | https://onebox.huawei.com/p/24add96997eb98344c8714c840ceafa9 |
| Totem | https://onebox.huawei.com/p/f392c318a3ec9f0fb002bc8bf0d16d75 |