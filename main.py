import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_selected(df, columns, start_index, end_index):
    """Plot the desired columns over index values in the given range."""
    plot_data(df.loc[start_index:end_index, columns], title="Selected Data")


def plot_data(df, title="Stock prices"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()


def normalize_data(df):
    return df / df.iloc[0]


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


def test_run():
    """Function called by Test Run."""
    # Define date range
    dates = pd.date_range('2010-01-01', '2010-12-31', name='Date')

    # Choose stock symbols to read
    symbols = ['GOOG', 'IBM', 'GLD']

    # Get stock data
    df = get_data(symbols, dates)
    df = normalize_data(df)
    plot_data(df, title="Stock Prices")

    # Slice and plot
    plot_selected(df, ['SPY', 'IBM'], '2010-3-1', '2010-4-1')


if __name__ == "__main__":
    test_run()
