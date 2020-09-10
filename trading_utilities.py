import os
import math
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

############################## Data Retrieval and Cleaning ##############################


def fill_missing_values(df):
    """Fill missing values in data frame, in place."""
    df.fillna(method='ffill', axis=0, inplace=True)
    df.fillna(method='bfill', axis=0, inplace=True)


def symbol_to_path(symbol, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)

    spyAdded = False
    if 'SPY' not in symbols:  # add SPY for trading day reference, if absent
        symbols.insert(0, 'SPY')
        spyAdded = True

    for symbol in symbols:
        # Read symbol data into temp dataframe, make sure to convert indicies to    date-time index objects
        dfTemp = pd.read_csv(symbol_to_path(symbol),
                             index_col="Date", parse_dates=True,
                             usecols=['Date', 'Adj Close'], na_values=['nan'])

        dfTemp = dfTemp.rename(columns={'Adj Close': symbol})
        df = df.join(dfTemp)
        if symbol == 'SPY':
            df = df.dropna(subset=['SPY'])
        fill_missing_values(df)

    if spyAdded:
        df = df.drop(['SPY'], axis=1)

    return df

############################ Data Plotting and Visualization ############################


def plot_selected(df, columns, start_index, end_index, title='Selected Data'):
    """Plot the desired columns over index values in the given range."""
    dfSelected = df.loc[start_index: end_index, columns]
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
    ha.set_xlabel("Date")
    ha.set_ylabel("Price")
    plt.show(block=False)
    plt.draw()
    plt.pause(0.01)
    return ha


############################### Operations on Dataframes ################################
def normalize_data(df):
    return df / df.iloc[0]


def compute_cumulative_returns(df):
    """Compute and return the cumulative return values."""
    # This is nearly the same as normalizing the data...
    return (df / df.iloc[0]) - 1


def compute_daily_returns(df):
    """Compute and return the daily return values."""
    # Note: Returned DataFrame must have the same number of rows
    daily_returns = df.copy()  # Copy to store results in
    # .values is necessary because pandas by default will try to
    # do element-wise operations based on index matching between dataframes.
    # .values returns the underlying numpy array which prevents pandas from
    # doing this.
    daily_returns[1:] = (df[1:] / df[: -1].values) - 1
    # OR
    # daily_returns = (df - df.shift(1)) - 1  # much easier
    # Set initial value to zero as there is no data prior
    #  to perform calculation
    daily_returns.iloc[0, :] = 0
    return daily_returns


def portfolio(symbols, dates, allocations, positionValue):
    """Calculate the sum portfolio value"""
    allocs = np.array(
        allocations)  # Convert to numpy so that element-wise operations are free
    assert(len(symbols) == len(allocations))
    df = get_data(symbols, dates)
    df = normalize_data(df)
    scale = allocs * positionValue
    df = df * scale
    df = df.sum(axis=1)
    return df


def compute_sharpe_ratio(df, samplingPeriod):
    dr = compute_daily_returns(df)
    dr = dr[1:]  # Remove the first zero
    k = 0
    if(samplingPeriod == 'daily'):
        k = math.sqrt(252)  # Num trading days
    elif (samplingPeriod == 'weekly'):
        k = math.sqrt(52)
    elif (samplingPeriod == 'monthly'):
        k = math.sqrt(12)
    else:
        sys.exit('Sampling period is incorrect')

    return k * dr.mean() / dr.std()
