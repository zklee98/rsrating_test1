from tqdm import tqdm
import pandas_ta as ta
import yfinance as yf
import pandas as pd
import numpy as np

import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

from zigzag import *
import time
import datetime as dt

from plot-day-chart import make_charts

csv_path = os.path.join(os.getenv('GITHUB_WORKSPACE'), 'output', 'rs_stocks.csv')

# Read the csv as dataframe then remain only ticker and rs rating
rs_stocks = pd.read_csv('/content/rs_stocks_25Feb.csv')
tickers_df = rs_stocks[['Ticker', 'Percentile']]

ticker_list = tickers_df['Ticker'].tolist()

make_charts(ticker_list)


