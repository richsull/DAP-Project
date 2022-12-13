"""
Musk Tweets - VADER Twitter Sentiment Analysis
Sentiment Analysis of posts on r/wallstreetbets subreddit: Jan 2022 - Oct 2022
Data set sourced through a Twitter API scrape. Python code source: https://www.youtube.com/watch?v=ujId4ipkBio&t=171s
Sentiment analysis is form of text analytics that uses natural language processing (NLP) and machine learning.
It is sometimes referred to as ‘opinion mining’. A key aspect of sentiment analysis is polarity classification.
Polarity refers to the overall sentiment conveyed by a particular tweet, phrase or word. Polarity can be expressed as a simple
numerical score (or rating). In this example, the score is represented as ‘below 0’, ‘zero’, or ‘above zero’, with zero representing a neutral sentiment
"""

# import packages
import urllib.request
import json
import datetime

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn
import psycopg2
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pprint import pprint
from IPython import display
plt.style.use("fivethirtyeight")
from sqlalchemy import create_engine

# Download vader lexicon if needed
print("Start of sentiment_tweet_vader script")

nltk.download('vader_lexicon')

# Define seaborn style
seaborn.set(style='darkgrid', context='talk', palette='Dark2')

# Create object of type SentimentIntensityAnalyzer
# creating object of SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Connect to Google Cloud postgres

username = 'project_users'  # DB username
password1 = 'DAPproject'  # DB password
host1 = '35.189.77.255'  # Public IP address for your instance
port1 = '5432'
database1 = 'postgres'  # Name of database ('postgres' by default)
dbConnection = psycopg2.connect(user=username,
                                password=password1,
                                host=host1,
                                port=port1,
                                database=database1
                               )

dbConnection.set_isolation_level(0) # Autocommit
dbCursor=dbConnection.cursor()
# define engine
engine = create_engine(f'postgresql://{username}:{password1}@{host1}:5432/{database1}')

# Query SQL table and create dataframe to store

query="SELECT * FROM tweets"
tweets_df = pd.read_sql(query, engine)
tweets_df = tweets_df.drop(['index'], axis=1)
#print(tweets_df.head())

"""
Calculate Sentiment for All Rows in a CSV
To calculate the sentiment for each tweet in the CSV file and add a new column that contains this information, 
we will create a function that will take in any text and output the compound sentiment score.
"""

def calculate_sentiment(text):
    # Run VADER on the text
    scores = sid.polarity_scores(text)
    # Extract the compound score
    compound_score = scores['compound']
    return compound_score

# Apply the function to every row in the "Comment" column and output  results into new column "Polarity"
tweets_df['Polarity'] = tweets_df['text'].apply(calculate_sentiment)

# Sort the data frame and examine the top 10 tweets with the highest compound sentiment.
tweets_df.sort_values(by='Polarity', ascending=False)[:10]

# Sort the data frame and examine the top 10 tweets with the lowest compound sentiment.
tweets_df.sort_values(by='Polarity', ascending=True)[:10]

# Create a function to store sentiment results in the data frame as Negative, Positive or Neutral
# create function to compute negative, positive an neutral analysis
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

# show the new dataframe
tweets_df['Analysis'] = tweets_df['Polarity'].apply(getAnalysis)

"""
### Plot Sentiment Over Time
We can plot how sentiment in the subreddit posts fluctuate over time by first converting the ‘time_created’ column to a 
datetime value and then making it the index of the data frame, which makes it easier to work with time series data."""

# Make date the index of the DataFrame
tweets_df['datetime'] = pd.to_datetime(tweets_df['datetime'])
tweets_df = tweets_df.set_index('datetime')

# Averages

# the mean of likecounts:
mean = tweets_df['likecount'].mean()

print("The average number of likes is: {:.2f}".format(mean))

# the mean of retweets:
mean = tweets_df['retweetcount'].mean()

print("The average number of retweets is: {:.2f}".format(mean))

"""
Now group the tweets by month using resample( ), a special method for datetime indices, and calculate the mean( ) 
compound score for each month. Finally, we will plot these averages.
"""

tweets_df.resample('M')['Polarity'].mean().plot(title="Polarity by Month")

# Resample by day (‘D’), week (‘W’), or year (‘Y’).

tweets_df.resample('W')['Polarity'].mean().plot(title="Polarity by Week")
tweets_df.resample('D')['Polarity'].mean().plot(title="Polarity by Day")


# Using .loc, we can also zoom in on particular time periods
# Trying day before May 13th 2022, and 2 days after that date

tweets_df.loc["2022-05-12":"2022-05-15"].sort_values(by='Polarity')

# Trying day before Oct 28th 2022, and 2 days after that date
tweets_df.loc["2022-10-27":"2022-10-31"].sort_values(by='Polarity')

# Check the percentage of positive, neutral and negative tweets in the data set
# Get the percentage of positive tweets in the data set
pposts = tweets_df[tweets_df.Analysis == 'Positive']
pposts = pposts['text']
p = round( (pposts.shape[0] / tweets_df.shape[0]) *100, 1)
print("{}% positive sentiment".format(p))

# Get the percentage of negative tweets in the data set
nposts = tweets_df[tweets_df.Analysis == 'Negative']
nposts = nposts['text']
n = round( (nposts.shape[0] / tweets_df.shape[0]) *100, 1)
print("{}% negative sentiment".format(n))

# Get the percentage of neutral tweets in the data set
ntposts = tweets_df[tweets_df.Analysis == 'Neutral']
ntposts = ntposts['text']
n = round( (ntposts.shape[0] / tweets_df.shape[0]) *100, 1)
print("{}% neutral sentiment".format(n))

# Create a bar chart of sentiments
# show the value counts
tweets_df['Analysis'].value_counts()

# plot and visualise the counts
plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Counts')
tweets_df['Analysis'].value_counts().plot(kind='bar')
plt.show()

print("End of sentiment_tweet_vader script.")