# Volatility Analysis & Forecasting

A research-oriented volatility modeling framework for financial time series.  
Focuses on estimating, forecasting, and evaluating market risk using econometric and statistical methods commonly used in quantitative trading and risk management.

## Overview

This project studies the dynamics of asset return volatility and builds models to:

- measure realized and conditional volatility
- capture clustering and regime shifts
- forecast short-term risk
- evaluate out-of-sample predictive power

Designed for **research-to-trade consistency**, with strict time alignment and walk-forward evaluation to avoid look-ahead bias.

## Methods

- Rolling realized volatility
- EWMA / RiskMetrics
- GARCH / EGARCH family models
- Regime detection (e.g., HMM)
- Out-of-sample volatility forecasting
- Statistical validation (RMSE, QLIKE, likelihood metrics)

## Data

- Daily log returns
- Price series
- Optional benchmark / market index

All computations use strictly historical information at time *t* to predict *t+1*.

## Goals

- Understand volatility clustering and persistence  
- Improve risk forecasting  
- Support position sizing, VaR estimation, and strategy risk control  



## Disclaimer

This repository is for research and educational purposes only.  
Nothing here is financial advice, and performance in backtests does not guarantee future results.



UNAUTHORIZED USAGE OF THE CONTENTS IS PROHIBITED AND MAY RESULT IN LEGAL ACTIONS
## Notes

This repository is for research and educational purposes only.  
Backtests and forecasts do not guarantee future performance.
