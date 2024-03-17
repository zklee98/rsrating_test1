# usage: make_charts(one_ticker as stirng or list of string, number_days(bars)for_showing)

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
import os

def make_charts(stock_list, days = 260,INTERVAL='1d'):

    if type(stock_list) == type('string'):
        stock_list = [stock_list]

    
    #with PdfPages(f'dayCharts_{dt.date.today()}_{bin_number}of4.pdf') as pdf:
    with PdfPages(f'Watchlist_{dt.date.today()}.pdf') as pdf:

        for index, STOCK in enumerate(stock_list):
            
            for interval in ['1d', '1wk']:
                
                INDEX = '^GSPC' # SP500
                df_ohlc_index = yf.Ticker(INDEX).history(actions = False, period = 'max', interval = str(interval), rounding=True )
                
                try:

                    info = yf.Ticker(STOCK).info


                    df = yf.Ticker(STOCK).history(actions = False, period = 'max', interval = str(interval), rounding=True )


                    df=df.join(df_ohlc_index['Close'], rsuffix='_idx')


                    # add TA columns
                    try:
                        df.ta.sma(length=50,append=True)
                        df.ta.sma(length=200,append=True)
                        df.ta.sma(length=150,append=True)
                        #df.ta.ema(length=21,append=True)
                        df['VOL_50'] = df.ta.sma(close=df['Volume'], length=50)

                    except Exception as e:
                        print(f"An error occurred: {e}")

                    # add RS-line column

                    stock_close = df['Close']
                    ind_close = df['Close_idx']
                    df['rs_line'] = stock_close/ind_close*100 * stock_close.shift(60)/(stock_close/ind_close *100).shift(60)*0.68  # 0.68 for placing like in  IBD chart


                    # add Max and Min extremum for price labels

                    #treshold = 0.05 # % of changing price to count as extremum

                    #df['max']=df.High[peak_valley_pivots(np.array(df.High), treshold, -treshold) == 1]
                    #df['min']=df.Low[peak_valley_pivots(np.array(df.Low), treshold, -treshold) == -1]


                    # last OHLC data for titles
                    o = df.Open[-1]
                    h = df.High[-1]
                    l = df.Low[-1]
                    c = df.Close[-1]
                    v = df.Volume[-1]
                    chg = (df.Close[-1]/df.Close[-2]-1)*100

                    # add space to the right side of chart
                    dfpad = df.tail(round(days/50)).copy()
                    dfpad.loc[:,:] = float('nan')
                    newdf = df.append(dfpad)
                    df = newdf



                    # styling chart

                    #mc = mpf.make_marketcolors(up='#2A3FE5', down='#DB39AD', inherit=True,vcdopcod=True)
                    mc = mpf.make_marketcolors(up='#2736E9', down='#DE32AE', inherit=True)

                    base_style = {  'axes.titlesize':       5,
                                    'axes.labelsize':       5,
                                    'lines.linewidth':      3,
                                    'lines.markersize':     4,
                                    'ytick.left':           False,
                                    'ytick.right':          True,
                                    'ytick.labelleft':      False,
                                    'ytick.labelright':     True,
                                    'xtick.labelsize':      5,
                                    'ytick.labelsize':      5,
                                    'axes.linewidth':         0.8,
                                    'savefig.pad_inches': 0.1,
                                    'savefig.bbox': 'tight',
                                    'grid.alpha':           0.2}

                    #ibd = mpf.make_mpf_style(marketcolors=mc, mavcolors=['green', 'red', 'black', 'blue'], y_on_right=True, rc=base_style)
                    ibd = mpf.make_mpf_style(marketcolors=mc, y_on_right=True, rc=base_style)

                    ### !!! temporary replace to chunks loop
                    df_slice = df[-days:]
                    
                    if (interval == '1d') and len(df_slice) < 260:
                        num_empty_rows = 260 - len(df_slice) 
                        empty_df = pd.DataFrame(index=pd.date_range(start=df_slice.index.min(), periods=num_empty_rows, freq='D'), columns = df_slice.columns)
                        df_slice = pd.concat([empty_df, df_slice])
                    
                    elif (interval == '1wk') and len(df_slice) < 54:
                        num_empty_rows = 54 - len(df_slice)
                        empty_df = pd.DataFrame(index=pd.date_range(start=df_slice.index.min(), periods=num_empty_rows, freq='D'), columns = df_slice.columns)
                        df_slice = pd.concat([empty_df, df_slice])

                    # ======== starting ploting ==================================================


                    # making grid of axis
                    egrid = (21,29)

                    fig = mpf.figure(style=ibd, figsize=(11, 8))
                    ax1 = plt.subplot2grid(egrid,(1,0),colspan=25,rowspan=16)
                    ax3 = plt.subplot2grid(egrid,(0,0),colspan=25,rowspan=1)
                    ax2 = plt.subplot2grid(egrid,(17,0),colspan=25,rowspan=4,sharex=ax1)



                    # remove gaps among axis panels
                    fig.tight_layout(h_pad= -1.6)

                    #  Set locator intervals

                    try:
                        lim_bottom =min(df_slice['rs_line'].min(), df_slice['SMA_200'].min())
                        lim_top =  max(df_slice['rs_line'].max(), df_slice['SMA_200'].max(),df_slice['High'].max())

                    except Exception as e:
                        lim_bottom =(df_slice['Low'].min())
                        lim_top =  (df_slice['High'].max())
            

                    #  Enable minors ticks visible:

                    ax1.minorticks_on()
                    ax2.minorticks_on()


                    ax1.grid(which='major',color='k')
                    ax1.grid(which='minor',color='gray')
                    ax2.grid(which='major',color='k')
                    ax2.grid(which='minor',color='gray')

                    ax2.grid(which='major',axis='y', alpha=0.04)

                    ax2.tick_params(axis='x', which='major', pad = 8)

                    ax1.tick_params(axis='x', which='both',labelbottom= False, labeltop=False )
                    ax3.tick_params(which = 'both',labelbottom= False, labeltop=False, labelright = False, bottom=False, top=False, right = False)

                    base = len(df_slice)
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
                    ax1.xaxis.set_major_locator(mticker.IndexLocator(base=base/10, offset=0))
                    ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))
                    ax1.xaxis.set_minor_locator(mticker.IndexLocator(base=base/50, offset=0))


                    # making y-axis locator align with prices scale

                    maxprice = df_slice.High.max()
                    if maxprice > 100:
                        locator_size = 100
                    elif maxprice < 10:
                        locator_size = 1
                    else:
                        locator_size = 10

                    #ax2.ticklabel_format(axis='y',style='plain')

                    ax1.yaxis.set_major_locator(mticker.MultipleLocator(locator_size))
                    ax1.yaxis.set_minor_locator(mticker.MultipleLocator(locator_size/4))


                    factor =  (max(df_slice.High) / max(df_slice['Close_idx'])) * 1/2.5
                    shift = max(df_slice.High)/1.5

                    sp = mpf.make_addplot( (df_slice['Close_idx']  * factor + shift ) , color='black', width=0.9, ax=ax1, alpha = 0.6)


                    try:
                        vol50 = mpf.make_addplot(
                            df_slice['VOL_50'], panel=2, color='red', width=0.6, ax=ax2)
                    except Exception as e:
                        print(f"An error occurred: {e}")

                    try:
                        rs_line = mpf.make_addplot(
                            df_slice['rs_line'], ax=ax1, color='blue', width=0.5, alpha=0.75, panel=1)
                    except Exception as e:
                        print(f"An error occurred: {e}")

                    try:
                        sma200 = mpf.make_addplot(
                            df_slice['SMA_200'], ax=ax1,color='black', width=0.5, panel=1)
                    except Exception as e:
                        print(f"An error occurred: {e}")

                    try:
                        sma50 = mpf.make_addplot(
                            df_slice['SMA_50'], ax=ax1,color='red', width=0.5, panel=1)
                    except Exception as e:
                        print(f"An error occurred: {e}")

                    try:
                        sma150 = mpf.make_addplot(
                            df_slice['SMA_150'], ax=ax1,color='orange', width=0.5,  panel=1)
                    except Exception as e:
                        print(f"An error occurred: {e}")


                    # text adding

                    kwargs = dict(horizontalalignment='center', color='#000000', fontsize = 4, backgroundcolor = 'white',
                                bbox=dict(boxstyle='square', fc='white', ec='none', pad=0))


                    # add price labels above/below extremum bars

                    #price_pad = (max(df_slice.High)-min(df_slice.High))*0.01

                    #for i in range(len(df_slice)):
                        #if not(np.isnan(df_slice['max'].iloc[i])):
                            #ax1.text(i+1, df_slice['max'].iloc[i]+price_pad, np.round(df_slice['max'].iloc[i],2), **kwargs, verticalalignment='bottom')
                        #if not(np.isnan(df_slice['min'].iloc[i])):
                            #ax1.text(i+1, df_slice['min'].iloc[i]-price_pad, np.round(df_slice['min'].iloc[i],2), **kwargs, verticalalignment='top')

                    

                    try:
                        mpf.plot(df_slice, ax=ax1, volume=ax2, addplot=[sp, rs_line, sma200,  sma50, sma150, vol50], datetime_format="%b'%y",tight_layout=True, xrotation=0,
                            scale_width_adjustment=dict(volume=0.3),ylim=(lim_bottom*0.1,lim_top*1.1), update_width_config=dict(ohlc_ticksize=0.5, ohlc_linewidth=0.85))

                    except Exception as e:
                        mpf.plot(df_slice, ax=ax1, volume=ax2, addplot=[sp], datetime_format="%b'%y",tight_layout=True, xrotation=0,
                            scale_width_adjustment=dict(volume=0.3), ylim=(lim_bottom*0.1, lim_top*1.1), update_width_config=dict(ohlc_ticksize=0.5, ohlc_linewidth=0.85))


                    try:
                        high_p = round((c/info['fiftyTwoWeekHigh']-1)*100)
                        low_p = round((c/info['fiftyTwoWeekLow']-1)*100)
                        ax3.text(0.01, 0.6, f"{info['shortName'].title()} / {info['symbol']}  ({info['sector']}  -  {info['industry']}) ", fontsize=8)
                    except:
                        ax3.text(0.01, 0.6, f" No data  ", fontsize=8)
                    try:
                        ax3.text(0.65, 0.2, f"MarketCap: {np.round(info['marketCap']/1000000000,2)}B     Shares Outstanding: {np.round(info['sharesOutstanding']/1000000,2)}M" , fontsize=8)
                    except:
                        ax3.text(0.65, 0.2, f" No data  ", fontsize=8)
                    try:
                        ax3.text(0.65, 0.6, f"52w High: {high_p}% ({info['fiftyTwoWeekHigh']})   52w Low: {low_p}% ({info['fiftyTwoWeekLow']})" , fontsize=8)
                    except:
                        ax3.text(0.8, 0.6, f" No data  ", fontsize=8)
                    try:
                        ax3.text(0.01, 0.2, f"Last Open: {o} High: {h} Low: {l} Close: {c} %Chg: {np.round(chg,2)}    Volume: {np.round(v/1000000,2)}M    {info['quoteType']}", fontsize=8)
                    except:
                        ax3.text(0.01, 0.2, f" No data  ", fontsize=8)


                    #enable volume log scale
                    ax2.set_yscale('symlog')
                    #ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: str(np.round(x/1000000,1))+'M'))
                    #ax2.yaxis.set_major_locator(mticker.LogLocator(base=5.0))


                    # add 100 to bottom of y-axis to put some useful info later
                    ylimit_ax1 = ax1.get_ylim()
                    ax1.set_ylim(ymin = ylimit_ax1[0]-ylimit_ax1[1]*0.1)

                    # Input ticker and number of ticker here
                    interval_title = 'Day Chart' if str(interval) == '1d' else 'Week Chart'
                    ax3.set_title(f'{STOCK} - [{index+1} of {len(watchlist_df)}]')

                    legend_properties = {'weight':'bold', 'size': 8}
                    #rs = 50
                    ax1.legend([f'{STOCK.upper()} ({interval_title})       RS Rating - {stocks_output_df.loc[stocks_output_df['Ticker'] == STOCK.upper()].Percentile.values[0]}'], 
                               prop=legend_properties, labelcolor='blue', handlelength = 0, loc='upper center')
                    #ax1.legend([f'[{tickers_df.iloc[index, 0]}] RS Rating - {tickers_df.iloc[index, 1]}'], prop=legend_properties, labelcolor='blue', handlelength = 0, loc='upper center')
                    


                    #plt.show(fig)
                    pdf.savefig()
                    passed_tickers.append(STOCK)
                    plt.close()



                except Exception as exception:
                    print('Problem with : ', STOCK)
                    print(exception)
                    failed_tickers.append(STOCK)
                    plt.close()


    
csv_path = os.path.join(os.getenv('GITHUB_WORKSPACE'), 'watchlist', 'watchlist.csv')
output_path = os.path.join(os.getenv('GITHUB_WORKSPACE'), 'rsrating_output', 'output', 'rs_stocks.csv')

# Read stocks output csv
stocks_output_df = pd.read_csv(output_path)
stocks_output_df = stocks_output_df[['Ticker', 'Percentile']]


# Read the watchlist as dataframe then remain only ticker and rs rating
watchlist_df = pd.read_csv(csv_path)
ticker_list = watchlist_df['Watchlist_Ticker'].tolist()

# Creat empty list for passsed & failed tickers
passed_tickers = []
failed_tickers = []

# Split jobs into 4 bin
#bin_size, remainder = divmod(len(rs_stocks), 4)

# Plot chart for each bin
#make_charts(ticker_list[:bin_size], bin_number =1) #Frist 25%
#make_charts(ticker_list[bin_size: bin_size+bin_size], bin_number=2)  #Second 25%
#make_charts(ticker_list[(bin_size*2): (bin_size*3)], bin_number=3)    #Third 25%
#make_charts(ticker_list[(bin_size*3): ((bin_size*4)+remainder)], bin_number=4)    #Fourth 25%

make_charts(ticker_list)


# Write the list of failed tickers to a text file

with open('Summary.txt', 'w') as file:
    file.write(f"Number of Tickers successful plotted: {len(passed_tickers)}")
    file.write(f"\nNumber of Failed Tickers: {len(failed_tickers)} out of {len(watchlist_df)}")
    file.write("\n\nFailed Tickers:\n")
    file.write("\n".join(failed_tickers))
    
print(f"Total tickers processed: {len(passed_tickers)}")
print(f"Failed tickers: {len(failed_tickers)}")


print('After plotting')
print("Current Working Directory:", os.getcwd())

# List the contents of the current working directory
print("Contents of Current Working Directory:")
for item in os.listdir():
    print(item)
