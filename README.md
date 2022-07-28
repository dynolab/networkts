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