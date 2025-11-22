# Advanced Time Series Forecasting with Neural Networks

## Overview

This project implements a deep learning system for time series forecasting with uncertainty quantification. It compares a neural network model with a statistical baseline to demonstrate significant improvements in prediction accuracy while maintaining well-calibrated uncertainty estimates.

## What This Project Does

The project predicts future values in time series data using two methods:
1. A multi-layer neural network with three hidden layers
2. A triple exponential smoothing baseline model

Both models generate 95% confidence intervals for predictions, allowing users to understand the reliability of forecasts.

## Key Results

- Neural Network RMSE: 0.8715
- Baseline RMSE: 5.6498
- Accuracy Improvement: 84.6%
- Coverage Probability: 94.90% (both models)
- Mean Interval Width: 3.2499 (Neural Network) vs 14.9060 (Baseline)

## System Requirements
