import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import threading

plts = []


def plot_selected(df, columns, start_index, end_index, title='Selected Data'):
    """Plot the desired columns over index values in the given range."""
    dfSelected = df.loc[start_index:end_index, columns]
    return plot_data(dfSelected, title=title), dfSelected


def add_plot_data(ax, df, label="Additional Data"):
    ax.plot(df, label=label)
    plt.legend()
    plt.show(block=False)
    plt.draw()
    plt.pause(0.01)


def plot_data(df, title="Stock prices"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    hf, ha = plt.subplots(1, 1)
    df.plot(title=title, fontsize=12, ax=ha)
    print(id(hf), id(ha))
    ha.set_xlabel("Date")
    ha.set_ylabel("Price")
    plt.show(block=False)
    plt.draw()
    plt.pause(0.01)
    return ha


def normalize_data(df):
    return df / df.iloc[0]


def compute_cumulative_returns(df):
    """Compute and return the cumulative return values."""
    # This is nearly the same as normalizing the data...
    return (df / df.iloc[0]) - 1


def symbol_to_path(symbol, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        # Read symbol data into temp dataframe, make sure to convert indicies to    date-time index objects
        dfTemp = pd.read_csv(symbol_to_path(symbol),
                             index_col="Date", parse_dates=True,
                             usecols=['Date', 'Adj Close'], na_values=['nan'])

        dfTemp = dfTemp.rename(columns={'Adj Close': symbol})
        df = df.join(dfTemp)
        if symbol == 'SPY':
            df = df.dropna(subset=['SPY'])

    return df


def compute_daily_returns(df):
    """Compute and return the daily return values."""
    # Note: Returned DataFrame must have the same number of rows
    daily_returns = df.copy()  # Copy to store results in
    # .values is necessary because pandas by default will try to
    # do element-wise operations based on index matching between dataframes.
    # .values returns the underlying numpy array which prevents pandas from
    # doing this.
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    # OR
    # daily_returns = (df - df.shift(1)) - 1  # much easier
    # Set initial value to zero as there is no data prior
    #  to perform calculation
    daily_returns.iloc[0, :] = 0
    return daily_returns


def get_final_input():
    print("executing!")
    input("\n\nPress the enter key to exit.")
    plt.close(fig='all')


def test_run():
    """Function called by Test Run."""
    # Define date range
    dates = pd.date_range('2010-01-01', '2012-12-31', name='Date')

    # Choose stock symbols to read
    symbols = ['GOOG', 'IBM', 'GLD']

    # Get stock data
    df = get_data(symbols, dates)
    df = normalize_data(df)
    plot_data(df, title="Normalized Stock Prices")

    ##################### Plot Subset of Stock Data ##################
    # Slice and plot
    ax, dfOut = plot_selected(df, ['SPY'], '2012-1-1', '2012-12-31')

    ##################### Bollinger Bands ############################
    # Add moving average to the plot
    mean = dfOut.rolling(window=20).mean()
    add_plot_data(ax, mean, label='Moving Average')

    # Add Bollinger Bands to plot
    rollStd = dfOut.rolling(window=20).std()
    upperBand = mean + 2 * rollStd
    lowerBand = mean - 2 * rollStd
    add_plot_data(ax, upperBand, label='Upper Band')
    add_plot_data(ax, lowerBand, label='Lower Band')

    ##################### Daily Returns ############################
    plot_data(compute_daily_returns(
        df.loc['2012-1-1':'2012-12-31', 'SPY']), "Daily Returns")

    plot_data(compute_cumulative_returns(
        df.loc['2012-1-1':'2012-12-31', 'SPY']), "Cumulative Returns")

    # start = time.time()
    # while(1):
    #     now = time.time()
    #     if (now - start) >= 1/60.0:
    #         print('Updating plot, time({})!'.format(now))
    #         start = now
    #         plt.pause(0.001)


if __name__ == "__main__":
    test_run()
    t = threading.Thread(target=get_final_input)
    t.start()
    plt.show()
