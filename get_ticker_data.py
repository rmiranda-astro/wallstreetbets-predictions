import pandas as pd
import yfinance as yf


def get_ticker_data(symbol, start_date, end_date):

    """
    Get daily ticker data for 'symbol' between 'start_date' and 'end date', return as a
    pandas dataframe.
    """

    ticker = yf.Ticker(symbol)

    df = ticker.history(period="1d", start=start_date, end=end_date)
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    return df


if __name__ == "__main__":

    # Get SPY daily ticker data between September 30, 2019 and October 1, 2020

    symbol = "SPY"
    df = get_ticker_data(symbol, "2019-09-30", "2020-10-1")
    df.to_csv("%s.csv" % symbol.lower())
