
import time
import threading
import numpy as np
import pandas as pd
import trading_utilities as tu
import matplotlib.pyplot as plt


def get_final_input():
    input("\n\nPress the enter key to exit.")
    plt.close(fig='all')


def test_run():
    """Function called by Test Run."""
    # Define date range
    dates = pd.date_range('1981-07-30', '2020-08-31', name='Date')

    # Choose stock symbols to read
    symbols = ['SPY', 'AAPL', 'GLD']

    # Get stock data
    df = tu.get_data(symbols, dates)
    # df = normalize_data(df)
    tu.plot_data(df, title="Stock Prices")

    ##################### Plot Subset of Stock Data ##################
    # Slice and plot
    ax, dfOut = tu.plot_selected(df, ['SPY'], '1981-07-30', '2020-08-31')

    ##################### Bollinger Bands ############################
    # Add moving average to the plot
    mean = dfOut.rolling(window=20).mean()
    tu.add_plot_data(ax, mean, label='20-Day Moving Average')

    # Add Bollinger Bands to plot
    rollStd = dfOut.rolling(window=20).std()
    upperBand = mean + 2 * rollStd
    lowerBand = mean - 2 * rollStd
    tu.add_plot_data(ax, upperBand, label='Upper Band')
    tu.add_plot_data(ax, lowerBand, label='Lower Band')

    ##################### Daily Returns ############################
    dr = tu.compute_daily_returns(
        df.loc['2012-1-1':'2012-12-31', ['SPY', 'AAPL']])
    fig2 = plt.figure()
    mean = dr['SPY'].mean()
    print("Mean:\n{}".format(mean))
    std = dr['SPY'].std()
    print("STD:\n{}".format(std))
    print("Kurtosis:\n{}".format(dr['SPY'].kurtosis()))
    # plot_data(dr, "Daily Returns")
    dr['AAPL'].hist(bins=20, label='AAPL')
    axs = dr['SPY'].hist(bins=20, label='SPY')
    axs.axvline(mean, color='w', linestyle='dashed', linewidth=2)
    axs.axvline(std, color='r', linestyle='dashed', linewidth=2)
    axs.axvline(-std, color='r', linestyle='dashed', linewidth=2)
    plt.legend(loc='upper right')
    plt.show(block=False)
    plt.draw()
    plt.pause(0.01)

    ##################### Cumulative Returns ############################
    tu.plot_data(tu.compute_cumulative_returns(
        df.loc['2012-1-1':'2012-12-31', 'SPY']), "Cumulative Returns")

    ##################### Scatter Plots ############################
    daily_returns = tu.compute_daily_returns(
        df.loc['2012-1-1':'2012-12-31', ['SPY', 'AAPL', 'GLD']])
    daily_returns.plot(kind='scatter', x='SPY', y='AAPL')
    beta, alpha = np.polyfit(daily_returns['SPY'], daily_returns['AAPL'], 1)
    plt.plot(daily_returns['SPY'], beta *
             daily_returns['SPY'] + alpha, '-', color='r')
    plt.show(block=False)
    plt.draw()
    plt.pause(0.01)
    daily_returns.plot(kind='scatter', x='SPY', y='GLD')
    beta, alpha = np.polyfit(daily_returns['SPY'], daily_returns['GLD'], 1)
    plt.plot(daily_returns['GLD'], beta *
             daily_returns['GLD'] + alpha, '-', color='r')
    plt.show(block=False)
    plt.draw()
    plt.pause(0.01)
    print(daily_returns.corr(method='pearson'))

    ##################### Sharpe Ratio ############################
    faang = ['FB', 'AAPL', 'AMZN', 'NFLX', 'GOOG']
    dates = pd.date_range('1997-01-01', '2020-01-01', name='Date')
    dfPort = tu.portfolio(faang, dates, [0.2] * 5, 1e6)
    tu.plot_data(dfPort, title='Portfolio Value')
    print("Sharpe Ratio: {}\n".format(
        tu.compute_sharpe_ratio(dfPort, samplingPeriod='daily')))


if __name__ == "__main__":
    test_run()
    t = threading.Thread(target=get_final_input)
    t.start()
    plt.show()
