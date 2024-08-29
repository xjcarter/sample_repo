# Backtesting Framework

This repository provides a comprehensive framework for backtesting trading strategies using Python. The collection of Python scripts and associated files allow users to define, test, and analyze trading strategies using historical market data.

## Repository Structure

### Core Components

1. **`backtest.py`**
   - **Description**: This is the base class used to develop new strategies for backtesting. It provides the fundamental structure and methods required for executing backtests, such as running the test loop, managing trades, and calculating performance metrics.
   - **Role**: It serves as the backbone of the framework, and any new strategy should inherit from this class to ensure consistency and reusability across different tests.

2. **`security.py`**
   - **Description**: This file defines the `Security` class, which represents a tradeable financial instrument. It includes attributes such as the security's symbol, price data, and other relevant properties necessary for performing trades within a backtest.
   - **Role**: The `Security` class is used to encapsulate the data and operations related to individual securities, which can then be applied to backtesting strategies.

3. **`portfolio.py`**
   - **Description**: This file contains the `Portfolio` class, which allows for the combination of multiple strategies into a single backtest. It manages the allocation of capital across different strategies and evaluates the overall performance of the portfolio.
   - **Role**: The `Portfolio` class facilitates the testing of multiple strategies together, providing insights into how they perform collectively, as opposed to individually.

4. **`run_backtest.sh`**
   - **Description**: A shell script designed to streamline the execution of individual backtests. It creates separate directories for each backtest run, helping to compartmentalize and organize the output files.
   - **Role**: This script aids in automating and managing the backtesting process, ensuring that each run is isolated and easy to review.

5. **`spy_run.py`**
   - **Description**: This file is the configuration script for running a specific backtest. It defines which securities to trade (e.g., SPY) and which backtest object (e.g., `Trend_Strategy`) to activate.
   - **Role**: It acts as a setup script for initializing and running a particular backtest, specifying all necessary parameters and components.

6. **`indicators.py`**
   - **Description**: A library of technical analysis indicators used within the backtesting framework. It processes market data to generate signals and analytics that can be used by trading strategies.
   - **Role**: The indicators in this library are essential for creating and executing strategies, as they provide the tools needed to interpret market data and make trading decisions.

### Supporting Components

7. **`buy_and_hold.py`**
   - **Description**: Implements a basic "Buy and Hold" strategy for benchmarking purposes. This strategy simply buys a security and holds it for the duration of the backtest, providing a baseline for comparison against other strategies.

8. **`spy_buy_and_hold_stats.py`**
   - **Description**: A script that runs the "Buy and Hold" strategy specifically on SPY and generates statistics on its performance. Useful for understanding the performance of SPY under a passive investment strategy.

9. **`spy_qqq_combined.py`**
   - **Description**: A script designed to test the combination of SPY and QQQ strategies, offering insights into the performance when these two are combined in a portfolio.

10. **`trend_strategy.py`**
    - **Description**: Implements a trend-following strategy, which seeks to capitalize on market trends by entering positions in the direction of the prevailing trend.

### Data Files

11. **`SPY.csv` and `QQQ.csv`**
    - **Description**: These are CSV files containing historical price data for SPY (S&P 500 ETF) and QQQ (Nasdaq-100 ETF), which are used as input data for running the backtests.
    - **Role**: The data in these files is consumed by the backtesting scripts to simulate trading and evaluate strategy performance over historical periods.

### Reporting and Analysis

12. **Additional Python Scripts**
    - **Description**: Various other Python scripts included in the repository are responsible for generating reports, visualizing data, and providing summary statistics of the backtest results.
    - **Role**: These scripts ensure that the output of the backtests is organized and presented in a readable and interpretable manner, making it easier to analyze the results.

## Getting Started

To start using this framework, follow these steps:

1. Clone the repository to your local machine.
2. Ensure that you have all the necessary dependencies installed. The primary dependencies include `pandas`, `numpy`, and `matplotlib`, among others.
3. Configure your backtest by editing the `spy_run.py` file to specify the securities and strategies you want to test.
4. Run the backtest using the `run_backtest.sh` script to execute your configurations and generate results in separate directories.
5. Analyze the results using the provided reporting scripts to understand the performance of your strategies.

