# Python script to extract stock prices from the AlphaVantage API

# Import packages

import csv
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import IPython
#database libraries
import psycopg2
from sqlalchemy import create_engine
import sqlalchemy
import pandas.io.sql as sqlio
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

print("Start of stock_database_0_92 script...")
# expand windows to allow scrolling
auto_scroll_threshold = 99999
pd.options.display.max_columns = 99999


# Alphavantage is a Y-combinator backed start-up providing free and priced premium financial data
# https://www.alphavantage.co/documentation/#

# Define API key
APIKEY="KA3WLMAG53VKX67"

# Create postgreSQL engine
username = 'project_users'  # DB username
password1 = 'DAPproject'  # DB password
host1 = '35.189.77.255'  # Public IP address for your instance
port1 = '5432'
database1 = 'postgres'  # Name of database ('postgres' by default)
engine = create_engine(f'postgresql://{username}:{password1}@{host1}:5432/{database1}')

# Define twitter ticket
TICKER='TWTR'

# API function
def download_api_data(input_range, download_df):
    for i in input_range:
        with requests.Session() as s:
            download = s.get(i)
            decoded_content = download.content.decode('utf-8')
            cr = csv.DictReader(decoded_content.splitlines(),
                                # fieldnames=['time','open','high','low','close','volume'],
                                delimiter=',')
            download_data = [row for row in cr]

            temp_df = clean_api_download(download_data)

            download_df = pd.concat([download_df, temp_df], axis=0)

            download_df = download_df.sort_index(axis=0)

    return download_df


def api_data(input_df):
    # This function provides a wrapper to set up the API calls
    # and adds delays to manage the API call limit of 5 per minute
    # download_api_data() extracts the data
    # clean_api_download() cleans the data
    #
    # Define the dates for the API, going backwards from today
    # Year=1 or 2, month = 1 to 11, each slice is __yearXmonthY__

    CSV_URL = []

    year = ['year1', 'year2']
    months = ['month1', 'month2', 'month3', 'month4', 'month5', 'month6', 'month7', \
              'month8', 'month9', 'month10', 'month11', 'month12']

    for y in year:
        for m in months:
            CSV_URL.append("https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&outputsize=full&" \
                           + "apikey=" + APIKEY + "&" \
                           + "symbol=" + TICKER + "&" \
                           + "interval=1min&" \
                           + "slice=" + y + m)

    input_df = download_api_data([CSV_URL[1], CSV_URL[2], CSV_URL[3], CSV_URL[4], CSV_URL[5]], input_df)

    time.sleep(60)  # waitto avoid making more than 5 calls to the API in a minute

    input_df = download_api_data([CSV_URL[6], CSV_URL[7], CSV_URL[8], CSV_URL[9], CSV_URL[10]], input_df)

    time.sleep(60)  # waitto avoid making more than 5 calls to the API in a minute

    input_df = download_api_data([CSV_URL[11]], input_df)

    return (input_df)

# Clean the API data:

# convert time field to datetime type
# insert it into the dataframe
# set new datetime as the df index
# sort the rows on the datetime values in ascending order
# remove the previoius time field
# convert all the numbers to float
# insert the ticker as a new column

def clean_api_download(api_df):
    output_df = pd.DataFrame(api_df)
    output_df.dropna()
    time2 = pd.to_datetime(output_df['time'])
    output_df.insert(0, column='datetime', value=time2)
    output_df = output_df.set_index('datetime')
    output_df = output_df.drop(labels='time', axis=1)

    for i in ['open', 'high', 'low', 'close', 'volume']:
        output_df[i] = output_df[i].astype('float64')

    output_df.insert(0, 'ticker', TICKER)

    output_df = output_df.sort_index(axis=0)

    return output_df

# dataframe populated with the cleaned API data download.
# Re-run this definition to drop current value and start download again
api_download_df=pd.DataFrame()
api_download_df=api_data(api_download_df)

api_download_df

# Generate Returns, Std Dev, Volume and Range Measures

def add_returns_1min(input_df):
    returns_1min = ((input_df['close'] / input_df['close'].shift(1)) - 1)

    # return for first observation cannot be calculated and 25 Dec and 1 Jan are included in the series from Alphavantage

    returns_1min.name = 'returns_1min'
    input_df = input_df.merge(returns_1min, how='outer', left_index=True, right_index=True, sort=True,
                              validate='one_to_one')

    # input_df['ticker']=TICKER

    return (input_df)

## Simple Returns

## This doesn't take account of the date differences, so that return could be open of day 2 v close of day 1

## returns_1min= ((stock_df['close'] / stock_df['close'].shift(1)) -1 )

def add_returns_1min(input_df):
    returns_1min = ((input_df['close'] / input_df['close'].shift(1)) - 1)

    # return for first observation cannot be calculated and 25 Dec and 1 Jan are included in the series from Alphavantage

    returns_1min.name = 'returns_1min'
    input_df = input_df.merge(returns_1min, how='outer', left_index=True, right_index=True, sort=True,
                              validate='one_to_one')

    # input_df['ticker']=TICKER

    return (input_df)

# 30 min returns

def add_returns_30min(input_df):
    ohlc_30min = input_df['close'].resample("30min", closed='right', label='right').ohlc()  # OHLC prices every 30 mins

    returns_30min = pd.Series(ohlc_30min['close'] / ohlc_30min['close'].shift(1)) - 1

    returns_30min.name = 'returns_30min'

    input_df = input_df.merge(returns_30min, how='outer', left_index=True, right_index=True, sort=True,
                              validate='one_to_one')

    input_df['ticker'] = TICKER

    return (input_df)

# 1 day returns

def add_returns_1d(input_df):
    ohlc_1d = input_df['close'].resample("B").ohlc()  # OHLC prices every business day
    ohlc_1d = ohlc_1d.dropna()  # 25 Dec and 1 Jan are included in the series from Alphavantage
    returns_1d = (ohlc_1d['close'] / ohlc_1d['close'].shift(
        1)) - 1  # calculate returns from today's close to yesteday's
    returns_1d.name = 'returns_1d'
    input_df = input_df.merge(returns_1d, how='outer', left_index=True, right_index=True, sort=True,
                              validate='one_to_one')
    input_df['ticker'] = TICKER

    return (input_df)

# 30 min volume

def add_volume_30min(input_df):
    volume_30min = input_df['volume'].resample("30min", closed='right',
                                               label='right').sum()  # sum of volume every business day

    volume_30min.name = 'volume_30min'

    input_df = input_df.merge(volume_30min, how='outer', left_index=True, right_index=True, sort=True,
                              validate='one_to_one')

    input_df['ticker'] = TICKER

    return (input_df)

# 1 day volume

def add_volume_1d(input_df):
    volume_1d = input_df['volume'].resample("B").sum()  # sum of volume every business day

    volume_1d.name = 'volume_1d'  # rename the column

    input_df = input_df.merge(volume_1d, how='outer', left_index=True, right_index=True, sort=True,
                              validate='one_to_one')

    input_df['ticker'] = TICKER

    return (input_df)

# 30 minute standard deviation

# Assuming 252 trading days in the year, we scale up from daily to annualised volatility by multiplying by the square root of 252,
# as standard e.g. Hull (2009, pp.282--285)

# volatility constants

print("sqrt(252)={}".format(np.sqrt(252)))  # assuming 252 business days in a year

print("sqrt(252*32)={}".format(
    np.sqrt(252 * 32)))  # assuming 16 hours of trading per business day, or 32 x 30 minute periods


def add_stdev_30min(input_df):
    stdev_30min = np.sqrt(252 * 32) * input_df['returns_1min'].resample("30min", closed='right', label='right').std()

    stdev_30min.name = 'stdev_30min'

    input_df['ticker'] = TICKER

    input_df = input_df.merge(stdev_30min, how='outer', left_index=True, right_index=True, sort=True,
                              validate='one_to_one')

    return (input_df)

# 1 day standard deviation

def add_stdev_1d(input_df):
    stdev_1d = np.sqrt(252) * input_df['returns_1min'].resample("B", closed='right', label='right').std()

    stdev_1d.name = 'stdev_1d'

    input_df = input_df.merge(stdev_1d, how='outer', left_index=True, right_index=True, sort=True,
                              validate='one_to_one')

    input_df['ticker'] = TICKER

    return (input_df)

# Range 1 min

def add_range_1min(input_df):
    high = input_df['high'].resample("1min", closed='right', label='right').max()  # OHLC prices every minute max
    low = input_df['low'].resample("1min", closed='right', label='right').min()  # OHLC prices every minute min

    range_1min = high - low

    range_1min = range_1min.dropna()  # get rid of the NA for any minute where there is no matching price

    range_1min.name = 'range_1min'

    input_df = input_df.merge(range_1min, how='outer', left_index=True, right_index=True, sort=True,
                              validate='one_to_one')

    input_df['ticker'] = TICKER

    return (input_df)

# Range 30 min

def add_range_30min(input_df):
    high = input_df['high'].resample("30min", closed='right', label='right').max()  # OHLC prices every business day

    low = input_df['low'].resample("30min", closed='right', label='right').min()  # OHLC prices every business day

    range_30min = high - low

    range_30min.name = 'range_30min'

    range_30min = range_30min.dropna()

    input_df = input_df.merge(range_30min, how='outer', left_index=True, right_index=True, sort=True,
                              validate='one_to_one')

    return (input_df)

# Range 1 day

def add_range_1d(input_df):
    high = input_df['high'].resample("B", closed='right', label='right').max()  # OHLC prices every business day

    low = input_df['low'].resample("B", closed='right', label='right').min()  # OHLC prices every business day

    range_1d = high - low

    range_1d.name = 'range_1d'

    range_1d = range_1d.dropna()

    input_df = input_df.merge(range_1d, how='outer', left_index=True, right_index=True, sort=True,
                              validate='one_to_one')

    return (input_df)

# Run transform of API data

def transform_cleaned_data(input_df):
    # add returns columns to the input dataframe
    input_df = add_returns_1min(input_df)
    input_df = add_returns_30min(input_df)
    input_df = add_returns_1d(input_df)

    # add extra volume calculation columns to the input dataframe
    input_df = add_volume_30min(input_df)
    input_df = add_volume_1d(input_df)

    # add standard deviations calculation columns to the input dataframe
    input_df = add_stdev_30min(input_df)
    input_df = add_stdev_1d(input_df)

    # add range columns to the input dataframe
    input_df = add_range_1min(input_df)
    input_df = add_range_30min(input_df)
    input_df = add_range_1d(input_df)

    return (input_df)

stock_df=transform_cleaned_data(api_download_df)

# Compare visually the cleaned API download to the transformed data in stock_df

api_download_df
stock_df

# Save database to csv

def save_df(dataframe, filename):
    from datetime import datetime
    current_date = str(datetime.now().year) + "_" + str(datetime.now().month) + "_" + str(datetime.now().day) \
                   + "_" + str(datetime.now().hour) + str(datetime.now().minute)
    outputfile = "filename_" + current_date + ".csv"
    print(outputfile)
    dataframe.to_csv(outputfile)

save_df(api_download_df,"download")
save_df(stock_df,"stock")

# Set Database Credentials

def create_cloud_db_credentials():
    user = 'project_users'  # DB username
    password = 'DAPproject'  # DB password
    host = '35.189.77.255'  # Public IP address for your instance
    port = '5432'
    database = 'postgres'  # Name of database ('postgres' by default)

    return user, password, host, port, database

# Database: create the db, table or drop them, query DB, upload or
# download cleaned or transformed data

# set credentials above for local or the cloud database before running this cell

def query_db(sqlString):
    user, password, host, port, database = create_cloud_db_credentials()

    import pandas.io.sql as sqli

    import psycopg2

    query_df = pd.DataFrame()

    try:
        dbConnection = psycopg2.connect(user=user,
                                        password=password,
                                        host=host,
                                        port=port,
                                        database=database
                                        )
        query_df = sqlio.read_sql_query(sqlString, engine)

    except (Exception, psycopg2.Error) as dbError:
        print("Error: ", dbError)

    finally:
        if (dbConnection):
            dbConnection.close()

    # print(database_download_df)

    return query_df


def select_all(table_name):
    sqlString = 'SELECT * FROM ' + table_name + ' ORDER BY datetime;'

    return sqlString

# Check if the database tables, stock_cleaned and stock_transformed exist in the target database

database_download_df=pd.DataFrame()
sqlString=select_all('stock_cleaned')
sqlString
database_download_df=query_db(sqlString)
database_download_df
database_download_df=database_download_df.set_index('datetime')
sqlString=select_all('stock_transformed')
sqlString
database_download_df=query_db(sqlString)
database_download_df=database_download_df.set_index('datetime')
database_download_df

# Upload stock data to SQL
def upload_stock_data(upload_df, upload_table_name):
    user, password, host, port, database = create_cloud_db_credentials()
    conn_string = 'postgresql://' + user + ":" + password + "@" + host + ":" + port + "/" + database
    db = create_engine(conn_string)
    conn = db.connect()
    conn.autocommit = True

    upload_df.to_sql(upload_table_name, db, if_exists='replace', index=True)

    conn.close()

upload_stock_data(api_download_df,'stock_cleaned')
api_download_df
upload_stock_data(stock_df,'stock_transformed')

## Download transformed stock data from database table stock_transformed
sqlString=select_all('stock_transformed')
sqlString
database_download_df=query_db(sqlString)
database_download_df=database_download_df.set_index('datetime')
stock_df=database_download_df
stock_df

## Database actions

def create_new_database(database_name):
    sqlString = 'CREATE DATABASE' + database_name + ';'

    return sqlString


def drop_table(table_name):
    # Drop a database table

    sqlString = 'DROP TABLE ' + table_name

    return sqlString


def create_cleaned_table(table_name):
    # schema for cleaned API stock data; the different schema below runs for processing for results after transform
    sqlString = 'CREATE TABLE ' + table_name + ' (datetime timestamp PRIMARY KEY, ticker varchar(8), \
    open real, high real, low real, close real, volume real);'''

    return sqlString


# schema for enriched stock data; the different schema above runs for processing cleaned data from the API

def create_transformed_table(table_name):
    sqlString = 'CREATE TABLE ' + table_name + ''' (
        datetime timestamp PRIMARY KEY,
        ticker varchar(8),
        open real,
        high real,
        low real,
        close real,
        volume real,
        returns_1min real,
        returns_30min real,
        returns_1d real,
        volume_30min real, 
        volume_1d real,
        stdev_30min real, 
        stdev_1d real,
        range_1min real,
        range_30min real,
        range_1d real
        );'''

    return sqlString


# set credentials above for local or the cloud database before running this cell

def call_database_string(sqlString):
    user, password, host, port, database = create_cloud_db_credentials()

    dbConnection = psycopg2.connect(user=user,
                                    password=password,
                                    host=host,
                                    port=port,
                                    database=database
                                    )
    dbConnection.set_isolation_level(0)  # Autocommit
    dbCursor = dbConnection.cursor()
    dbCursor.execute(sqlString)
    dbCursor.close()
    dbConnection.close()

sqlString=create_cleaned_table('stock_cleaned')
sqlString

sqlString=create_transformed_table('stock_transformed')
sqlString


## Event Tests

"""Do tweets have information value that investors will act on? Here, we test the events from the Financial Times
 timeline of @ElonMusk tweets about the Twitter takeover (https://www.ft.com/content/b0b49bc2-9d6e-4e0d-8962-a0f60185947c)"""


input_df=stock_df
input_df
input_df['stdev_30min']['2022-05-12 18:00':'2022-05-13 07:00'].dropna()[0:100]#['2022-04-03':'2022-04-03']
tweet_datetime='2022-05-13 05:44'

## Events tests functions

def events_tests(tweet_datetime, input_df, significance=0.05):
    # INPUT tweet datetime = string in format 'YYYY-MM-DD HH:MM' e.g. '2022-05-13 05:44'
    # OPTIONAL significance for statistical tests, set by default at 5%

    # Returns
    # 1. tweet_datetime, used for generating two samples, up to the tweet dateime, and then including and after

    # 2. t2_statistic = t-statistic from a Welch t-test for unequal variance samples before and after time

    # 3. t2_pvalue = probability of the null hyptothesis H_0 of both means being equal

    # 4. Result1= interpretation of the t-test is consistent with H_0 or H1, means are equal, not equal
    #    i.e. "Mean different" or "Mean same"

    # 5. levene_statistic = outcome of a Levene test for equality of variance in the two samples

    # 6. levene_pvalue, probability of the null hypothesis H_0 of equal variance, from the levene test
    # i.e. "Variance different" or "Variance same"

    # Test samples come from split into a pre-Tweet period (pre_start to pre_end)
    # and another starting from the Tweet datetime (post_start to post_end)
    # Each sample pre and post has 30 observations of the 30 minute statistic, so 15 hours of calendar time
    # test uses the stock returns_30mins variable statistic as input to the t- and levene tests

    # Tweet date and time is the start of the post-event testing
    post_start = pd.Timestamp(tweet_datetime)

    # Pre-Tweet observation period start date and time
    pre_start = post_start + pd.Timedelta(-15, unit="hours")

    # Pre-Tweet observation period end date and time
    pre_end = post_start + pd.Timedelta(-1, unit="minutes")

    # Post-Tweet observation period end date and time
    post_end = post_start + pd.Timedelta(15, unit="hours")

    # before = returns for the pre-Tweet period
    before = input_df['returns_30min'][pre_start:pre_end].dropna()

    # before = returns for the pre-Tweet period
    after = input_df['returns_30min'][post_start:post_end].dropna()

    # Calculate cumulative return before the tweet
    cum_return_before = pd.Series(data=np.ones(len(before)), index=before.index, dtype=float)

    # Convert the simple to cumulative returns
    for i in range(0, len(before)):
        cum_return_before[i] = 1 + before[i]
        if (i > 0):  cum_return_before[i] *= cum_return_before[i - 1]

    # Calculate cumulative return after the tweet
    cum_return_after = pd.Series(data=np.ones(len(after)), index=after.index, dtype=float)

    # Convert the simple to cumulative returns
    for j in range(0, len(after)):
        cum_return_after[j] = 1 + after[j]
        if (i > 0):  cum_return_after[j] *= cum_return_after[j - 1]

    # standard, assumes equal variance of samples, which is not a safe assumption
    t1 = stats.ttest_ind(cum_return_before, cum_return_after)

    # Welch test, which works with an unequal variance of the samples
    t2_statistic, t2_pvalue = stats.ttest_ind(cum_return_before, cum_return_after, equal_var=False,
                                              alternative='two-sided')

    # significance
    if (t2_pvalue < significance):
        result1 = "H_1: Mean different"
    else:
        result1 = "H_0: Mean same"

    levene_statistic, levene_pvalue = stats.levene(cum_return_before, cum_return_after, center='median')

    if (levene_pvalue < significance):
        result2 = "H_1: Variance different"
    else:
        result2 = "H_0: Variance same"

    return [tweet_datetime, t2_statistic, t2_pvalue, significance, result1, levene_statistic, levene_pvalue, result2]


def events_tests_graphs(tweet_datetime, input_df):
    # INPUT tweet datetime = string in format 'YYYY-MM-DD HH:MM' e.g. '2022-05-13 05:44'

    ## Graph the tweet date and time against
    # 1. price (close),
    # 2. returns over 30 mins (returns_30min),
    # 3. std dev of returns on 30 mins (stdev_30min)
    # 4. range over 30 mins (range_30mins)

    # Tweet date and time is the start of the post-event testing
    post_start = pd.Timestamp(tweet_datetime)

    # Pre-Tweet observation period start date and time
    pre_start = post_start + pd.Timedelta(-15, unit="hours")

    # Pre-Tweet observation period end date and time
    pre_end = post_start + pd.Timedelta(-1, unit="minutes")

    # Post-Tweet observation period end date and time
    post_end = post_start + pd.Timedelta(15, unit="hours")

    # before = returns for the pre-Tweet period
    before = input_df['returns_30min'][pre_start:pre_end].dropna()

    # before = returns for the pre-Tweet period
    after = input_df['returns_30min'][post_start:post_end].dropna()

    # Graph 1 - price
    p = input_df['close'][pre_start:post_end].dropna().plot(title=TICKER + " " + str(post_start.date()))
    # p.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    p.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.axvline(x=tweet_datetime, color='r', label='tweet')

    plt.axhline(y=0, linewidth=0.25, color='black')
    p.legend()
    plt.show()

    # Graph 2 - returns_30 mins
    p = input_df['returns_30min'][pre_start:post_end].dropna().plot(title=TICKER + " " + str(post_start.date()))
    p.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    p.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.axvline(x=tweet_datetime, color='r', label='tweet')
    plt.axhline(y=0, linewidth=0.25, color='black')
    p.legend()
    plt.show()

    # Graph 3 - stdev_30 mins
    p = input_df['stdev_30min'][pre_start:post_end].dropna().plot(title=TICKER + " " + str(post_start.date()))
    p.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    # p.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
    # p.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.axvline(x=tweet_datetime, color='r', label='tweet')
    plt.axhline(y=0, linewidth=0.25, color='black')
    p.legend()
    plt.show()

    # Graph 4 - range_30 mins
    p = input_df['range_30min'][pre_start:post_end].dropna().plot(title=TICKER + " " + str(post_start.date()))
    # p.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    p.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.axvline(x=tweet_datetime, color='r', label='tweet')
    plt.axhline(y=0, linewidth=0.25, color='black')
    p.legend()
    plt.show()

    # Graph 5 - volume_30 mins
    p = input_df['volume_30min'][pre_start:post_end].dropna().plot(title=TICKER + " " + str(post_start.date()))
    # p.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    # p.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.axvline(x=tweet_datetime, color='r', label='tweet')
    plt.axhline(y=0, linewidth=0.25, color='black')
    p.legend()
    plt.show()


### 13 May 2022 05:44
###  'Musk claims Twitter deal "on hold" because of spam bots and calls on Agrawal
###  to "prove" metrics on spam and bot accounts'

# Identify the date and time of the tweet event to be analysed
tweet_datetime='2022-05-13 05:44'

### 1. Price (close)
### 2. Returns over 30 mins (returns_30min),
### 3. Std dev of returns on 30 mins (stdev_30min)
### 4. Range over 30 mins (range_30mins)
### 5. Volume over 30 mins periods (volume_30mins)

print('tweet_datetime\n','t2_statistic\n', 't2_pvalue\n', 'significance\n', 'result1\n', 'levene_statistic\n',
      'levene_pvalue\n', 'result2\n')

events_tests(tweet_datetime, input_df, significance=0.05)
events_tests_graphs(tweet_datetime, input_df)

events_tests('2022-04-04 12:04',input_df, significance=0.05)
events_tests_graphs('2022-04-04 12:04',input_df)

print('tweet_datetime\n','t2_statistic\n', 't2_pvalue\n', 'significance\n', 'result1\n', 'levene_statistic\n',
      'levene_pvalue\n', 'result2\n')

events_tests('2022-04-14 07:23',input_df, significance=0.05)
events_tests_graphs('2022-04-14 07:23',input_df)

print('tweet_datetime\n','t2_statistic\n', 't2_pvalue\n', 'significance\n', 'result1\n', 'levene_statistic\n',
      'levene_pvalue\n', 'result2\n')

events_tests('2022-04-21 14:53',input_df, significance=0.05)
events_tests('2022-04-21 14:53',input_df, significance=0.05)

print('tweet_datetime\n','t2_statistic\n', 't2_pvalue\n', 'significance\n', 'result1\n', 'levene_statistic\n',
      'levene_pvalue\n', 'result2\n')

events_tests('2022-04-25 15:43',input_df, significance=0.05)
events_tests_graphs('2022-04-25 15:43',input_df)

print('tweet_datetime\n','t2_statistic\n', 't2_pvalue\n', 'significance\n', 'result1\n', 'levene_statistic\n',
      'levene_pvalue\n', 'result2\n')

events_tests('2022-05-05 09:14',input_df, significance=0.05)
events_tests_graphs('2022-05-05 09:14',input_df)

print('tweet_datetime\n','t2_statistic\n', 't2_pvalue\n', 'significance\n', 'result1\n', 'levene_statistic\n',
      'levene_pvalue\n', 'result2\n')

events_tests('2022-05-13 05:44',input_df, significance=0.05)
events_tests_graphs('2022-05-13 05:44',input_df)

print('tweet_datetime\n','t2_statistic\n', 't2_pvalue\n', 'significance\n', 'result1\n', 'levene_statistic\n',
      'levene_pvalue\n', 'result2\n')

events_tests('2022-10-04 18:39', input_df, significance=0.05)
events_tests_graphs('2022-10-04 18:39', input_df)

print('tweet_datetime\n','t2_statistic\n', 't2_pvalue\n', 'significance\n', 'result1\n', 'levene_statistic\n',
      'levene_pvalue\n', 'result2\n')

events_tests('2022-10-26 14:45',input_df, significance=0.05)
events_tests_graphs('2022-10-26 14:45',input_df)

print('tweet_datetime\n','t2_statistic\n', 't2_pvalue\n', 'significance\n', 'result1\n', 'levene_statistic\n',
      'levene_pvalue\n', 'result2\n')

events_tests('2022-10-27 09:08',input_df, significance=0.05)
events_tests_graphs('2022-10-27 09:08',input_df)

## Event tests summary table

tweet_times=[
'2022-3-25 02:34',
'2022-4-4 12:04',
'2022-4-14 07:23',
'2022-4-21 14:53',
'2022-4-25 15:43',
'2022-5-5 09:14',
'2022-5-13 05:44',
'2022-10-4 18:39',
'2022-10-26 14:45',
'2022-10-27 09:08'
]

t_tests_results= []

for i in range(0,len(tweet_times)):
    t_tests_results.append(events_tests(tweet_times[i], input_df))

events_tests_results=pd.DataFrame(columns=['tweet_datetime','t2_statistic', 't2_pvalue', 'significance', 'result1', \
                                     'levene_statistic', 'levene_pvalue', 'result2'])

for i in t_tests_results:
    events_tests_results.loc[len(events_tests_results)]=i

events_tests_results.style.set_properties(**{'text-align': 'left'})

# Testing Twitter and Reddit Sentiment

twitter_df=pd.read_csv('twitter_aggregates.csv')
# inspect visually
twitter_df

# Clean the Twitter polarity data file by adding datetime as an index and
# cutting to match the test dates of 5 April to 10 October 2022

twitter_dates=pd.to_datetime(pd.to_datetime(twitter_df['date']))
twitter_df.insert(0,column='datetime',value=twitter_dates)
twitter_df=twitter_df.set_index('datetime')
twitter_df=twitter_df['2022-04-05':'2022-10-27']

twitter_df

# Retrieve stock returns

sample_returns=stock_df['returns_1d']['2022-04-05':'2022-10-27'].dropna()
sample_returns
sample_returns.index

# Add stock returns to Twitter dataframe

twitter_df=twitter_df.merge(sample_returns,how='left',left_index=True,right_index=True,sort=True,validate='one_to_one')

# Retrieve daily 1min stock returns standard deviation

sample_stdev = stock_df['stdev_1d']['2022-04-05':'2022-10-27'].dropna()

# Add stock std dev to Twitter dataframe

twitter_df=twitter_df.merge(sample_stdev,how='left',left_index=True,right_index=True,sort=True,validate='one_to_one')
twitter_df

# Drop NA values, as many days did not have a tweet or Reddit post we could retrieve

twitter_df.isna().sum()
twitter_df=twitter_df.dropna()
twitter_df.isna().sum()
twitter_df

### Regress y= 1d stock returns against x= 1d Twitter mean polarity score

slope, intercept, r_value, p_value, std_err = \
stats.linregress(twitter_df['twitter_mean_1d'],twitter_df['returns_1d'])
print("r_value={:.4f}, p_value={:.4%}".format(r_value,p_value))
print("R squared = {:.4%}".format(r_value**2))
print("No significant R squared and probability value looks too high to be a rejection of the null hypothesis that the coefficients are all zero. We accept H_0, so there is likely no relationship.")

### Regress y=1d stock returns against x= 1d Twitter standard deviation of polarity score

slope, intercept, r_value, p_value, std_err = \
stats.linregress(twitter_df['twitter_stdev_1d'],twitter_df['returns_1d'])
print("r_value={:.4f}, p_value={:.4%}".format(r_value,p_value))
print("R squared = {:.4%}".format(r_value**2))
print("No significant R squared and probability value looks too high to be a rejection of the null hypothesis that the coefficients are all zero. We accept H_0, so there is likely no relationship.")

### Regress y=1d stock std dev against x = 1d Twitter standard deviation of polarity score
slope, intercept, r_value, p_value, std_err = \
stats.linregress(twitter_df['twitter_stdev_1d'],twitter_df['stdev_1d'])
print("r_value={:.4f}, p_value={:.4%}".format(r_value,p_value))
print("R squared = {:.4%}".format(r_value**2))
print("No significant R squared and probability value looks too high to be a rejection of the null hypothesis that the coefficients are all zero. We accept H_0, so there is likely no relationship.")

# Reddit sensitivity
reddit_df=pd.read_csv('reddit_aggregates.csv')
reddit_df=reddit_df.set_index('datetime')
reddit_df=reddit_df.dropna()
reddit_df

# Regress y=1d stock returns against x= 1d Reddit mean of polarity score

slope, intercept, r_value, p_value, std_err = \
stats.linregress(reddit_df['mean_pol_1d'],reddit_df['returns_1d'])
print("r_value={:.4f}, p_value={:.4%}".format(r_value,p_value))
print("R squared = {:.4%}".format(r_value**2))
print('No significant R squared and probability value looks too high to be a rejection of the null hypothesis that the coefficients are all zero. We accept H_0, so there is likely no relationship.')

# Regress y=1d stock returns against x= 1d Reddit standard deviation of polarity score
slope, intercept, r_value, p_value, std_err = \
stats.linregress(reddit_df['stdev_pol_1d'],reddit_df['returns_1d'])
print("r_value={:.4f}, p_value={:.4%}".format(r_value,p_value))
print("R squared = {:.4%}".format(r_value**2))
print('No significant R squared and probability value looks too high to be a rejection of the null hypothesis that the coefficients are all zero. We accept H_0, so there is likely no relationship.')

# Regress y=1d stock standard deviation of returns against x= 1d Reddit standard deviation of polarity score
slope, intercept, r_value, p_value, std_err = \
stats.linregress(reddit_df['stdev_pol_1d'],reddit_df['stdev_1d'])
print("r_value={:.4f}, p_value={:.4%}".format(r_value,p_value))
print("R squared = {:.4%}".format(r_value**2))
print('No significant R squared and probability value looks too high to be a rejection of the null hypothesis that the coefficients are all zero. We accept H_0, so there is likely no relationship.')


print("End of stock_database_0_92 script.")