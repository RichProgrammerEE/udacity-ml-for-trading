
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
    faang = ['AAPL', 'XOM', 'GLD', 'GOOG']
    dates = pd.date_range('2010-01-01', '2010-12-31', name='Date')
    dfPort = tu.portfolio(faang, dates, [0.2] * 4, 1)
    tu.plot_data(dfPort, title='Portfolio Value')
    print("Sharpe Ratio: {}\n".format(
        tu.compute_sharpe_ratio(dfPort, samplingPeriod='daily')))

    ##################### Line Optimization #######################
    # Plot the original line
    truth = np.float32([4, 2])
    print('Original line: C0 = {}, C1 = {}'.format(truth[0], truth[1]))
    Xtruth = np.linspace(0, 10, 21)
    Ytruth = truth[0] * Xtruth + truth[1]
    plt.figure()
    plt.plot(Xtruth, Ytruth, 'b--', linewidth=2.0, label='Original Value')

    # Generate noisy data points
    sigma = 3.0
    noise = np.random.normal(0, sigma, Ytruth.shape)
    data = np.asarray([Xtruth, Ytruth + noise]).T
    plt.plot(data[:, 0], data[:, 1], 'go', label='Data Points')

    # Try to fit a line to this data
    fit = tu.fit_line(data, tu.error)
    print('Fitted line: C0 = {}, C1 = {}'.format(fit[0], fit[1]))
    plt.plot(data[:, 0], fit[0] * data[:, 0] +
             fit[1], 'r--', linewidth=2.0, label='Fitted Line')

    # Add legend and show plot
    plt.legend()
    plt.show(block=False)
    plt.draw()
    plt.pause(0.01)

    ##################### Polynomial Optimization #################
    # Plot the original line
    poly = np.poly1d([1.5, -10, -5, 60, 50])
    print('Original Polynomial: {}'.format(poly))
    Xtruth = np.linspace(-5, 5, 21)
    Ytruth = np.polyval(poly, Xtruth)
    plt.figure()
    plt.plot(Xtruth, Ytruth, 'b--', linewidth=2.0, label='Original Polynomial')

    # Generate noisy data points
    sigma = 100.0
    noise = np.random.normal(0, sigma, Ytruth.shape)
    data = np.asarray([Xtruth, Ytruth + noise]).T
    plt.plot(data[:, 0], data[:, 1], 'go', label='Data Points')

    # Try to fit a line to this data
    fit = tu.fit_poly(data, tu.error_poly, degree=len(poly))
    print('Fitted Polynomial:{}'.format(fit))
    plt.plot(Xtruth, np.polyval(fit, Xtruth), 'r--',
             linewidth=2.0, label='Fitted Polynomial')

    # Add legend and show plot
    plt.legend()
    plt.show(block=False)
    plt.draw()
    plt.pause(0.01)


if __name__ == "__main__":
    test_run()
    t = threading.Thread(target=get_final_input)
    t.start()
    plt.show()
