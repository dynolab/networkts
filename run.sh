#!/bin/sh
nohup python -u examples/single_variable_forecasting.py +dataset=totem +decomposition=log +forecaster=autoreg > output/task.out 2> output/task.err < /dev/null &