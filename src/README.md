# SerinGrid.jl

**SPARE Estimation Regional Inventory Grids**

SerinGrid is a high-performance Julia package designed for advanced inventory management and spare parts forecasting. It implements a proprietary 12-grid classification matrix that synthesizes standard ABC revenue analysis with Coefficient of Variation (CV) volatility tracking. 

Designed for high-frequency data streams, SerinGrid quickly categorizes regional machine spares to feed seamlessly into larger predictive architectures (like DeepAR or hybrid ML models).

## Features
* **12-Grid SPARE Matrix:** Maps inventory across 3 revenue bands (A, B, C) and 4 volatility depths (V1, V2, V3, V4).
* **Dynamic Volatility Thresholds:** Customize the CV limits to match specific regional or seasonal behaviors.
* **Pareto Revenue Splits:** Built-in, adjustable 80/15/5 cumulative revenue calculations.
* **High-Performance:** Built on `DataFrames.jl` for lightning-fast, in-place matrix operations.

## Installation
From the Julia REPL, type `]` to enter the Pkg prompt and run:
```julia
pkg> add SerinGrid