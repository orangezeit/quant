# Stochastic Processes and Option Pricing

[![python_ver](https://img.shields.io/badge/python-3.8-brightgreen.svg)](https://www.python.org/)
[![numpy_ver](https://img.shields.io/badge/numpy-1.17-brightgreen.svg)](https://docs.scipy.org/doc/)
[![scipy_ver](https://img.shields.io/badge/scipy-1.4.1-brightgreen.svg)](https://docs.scipy.org/doc/)

## Option Types

* Call
* Put
* Call Spread
* Put Spread
* Call Binary
* Put Binary

## Style Types

* European
* American
* Bermudan

## Exercise Types

* Vanilla
* Exotic - Single Asset
  - Asian fixed
  * Asian float
  * Lookback fixed
  * Lookback float
  * Barrier
* Exotic - Multiple Assets (in development)
  * Basket min
  * Basket max
  * Atlas
  * Everest
  * Himalayan

## Model Types

### (General) Stochastic

* Geneator / Modifier
* Monte-Carlo
  - Weiner Process (Brownian Motion)
  - Jump-Diffusion
* Simulation and Option Pricing

### (Specific) Stochastic

* (1D / Interest Rate) Ornstein-Uhlenbeck
* (1D / Interest Rate) Cox-Intergell-Ross
* (1D / Stock Price) Constant Elasticity of Variance / CEV
  - PDE Scheme
  - Calibration
  - Bachelier / Normal Price Distribution
  - Black-Scholes / Log-Normal Price Distribution
    + Closed Form Formula for Price and Greeks of European option
    + Closed Form Formula for Price of Barrier option (in development)
* (2D / Stock Price) Heston
  - Calibration
* (2D / Stock Price) Stochastic-Alpha-Beta-Rho / SABR
  - Calibration
  
# Portfolio Management and Optimization

[![python_ver](https://img.shields.io/badge/python-3.8-brightgreen.svg)](https://www.python.org/)
[![numpy_ver](https://img.shields.io/badge/numpy-1.17-brightgreen.svg)](https://docs.scipy.org/doc/)
[![scipy_ver](https://img.shields.io/badge/scipy-1.4.1-brightgreen.svg)](https://docs.scipy.org/doc/)
[![pandas_ver](https://img.shields.io/badge/pandas-1.0.1-brightgreen.svg)](https://pandas.pydata.org/)
[![matplotlib_ver](https://img.shields.io/badge/matplotlib-3.2.1-brightgreen.svg)](https://matplotlib.org/)
[![sklearn_ver](https://img.shields.io/badge/sklearn-0.22.2-brightgreen.svg)](https://scikit-learn.org/stable/)

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
