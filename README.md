# Stochastic

[![python_ver](https://img.shields.io/badge/python-3.6-brightgreen.svg)](https://www.python.org/)
[![numpy_ver](https://img.shields.io/badge/numpy-1.17-brightgreen.svg)](https://docs.scipy.org/doc/)
[![scipy_ver](https://img.shields.io/badge/scipy-1.3-brightgreen.svg)](https://docs.scipy.org/doc/)

Stochastic Processes and Option Pricing

## Option Types

* Call
* Put
* Call Spread
* Put Spread
* Call Binary
* Put Binary

## Exercise Types

* Vanilla
  - European
  - American
* Exotic - Single Asset
  - Asian fixed
  * Asian float
  * Lookback fixed
  * Lookback float
  * Barrier
* Exotic - Multiple Assets
  * Basket min
  * Basket max
  * Atlas
  * Everest
  * Himalayan

## Model Types

### (General) Stochastic

* Geneator / Modifier
* Monte-Carlo
* Simulation and Option Pricing

### (Specific) Stochastic

* (1D / Interest Rate) Ornstein-Uhlenbeck
* (1D / Interest Rate) Cox-Intergell-Ross
* (1D / Stock Price) Constant Elasticity of Variance / CEV
  - PDE Scheme
  - Calibration
  - Bachelier / Normal Price Distribution
  - Black-Scholes / Log-Normal Price Distribution
    + Closed Form Formula for Price and Greeks
* (2D / Stock Price) Heston
  - Calibration
* (2D / Stock Price) Stochastic-Alpha-Beta-Rho / SABR
  - Calibration
  
# Portfolio

Portfolio Management and Optimization

## Dependency

* numpy
* pandas
* matplotlib
* scipy
* sklearn

## Estimation of Means and Covariances

* Mean
  - Equal Weights
  - Exponential Weights
* Covariances
  - Equal Weights
  - Exponential Weights
  - (Bayesian) Ledoit-Wolf
  - (Bayesian) OAS

## Markowitz

* Global Minimum Variance / GMV
* Maximum Sharpe Ratio / MSR
* Optimized Portfolio with Expected Returns
* Optimized Portfolio with Expected Risks
* Efficient Frontier Visualization
* Performance

## Black-Litterman

## Risk Parity

* Inverse Variance Portfolio
* Hierarchical Risk Parity
* Tail Risk Parity
