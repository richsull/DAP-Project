## WallStreetBets - VADER Reddit Sentiment Analysis

"""
Sentiment Analysis of posts on r/wallstreetbets subreddit: Jan 2022 - Oct 2022

Data set sourced through a Twitter API scrape. Python code source: https://www.youtube.com/watch?v=ujId4ipkBio&t=171s

Sentiment analysis is form of text analytics that uses natural language processing (NLP) and machine learning.
It is sometimes referred to as 'opinion mining'. A key aspect of sentiment analysis is polarity classification.
Polarity refers to the overall sentiment conveyed by a particular tweet, phrase or word.
Polarity can be expressed as a simple numerical score (or rating). In this example, the score is represented as
 'below 0', 'zero', or 'above zero', with zero representing a neutral sentiment.
"""

# import packages
import urllib.request
import json
import datetime
import psycopg2

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from sqlalchemy import create_engine
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pprint import pprint
from IPython import display

from collections import Counter
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
plt.style.use("fivethirtyeight")

"""
We now need to download the lexicon for NLTK, that will be used to analyze the words extracted from Reddit titles. 
VADER (Valence Aware Dictionary and sEntiment Reasoner) is used as the pre-trained model.
"""
print('Start of sentiment_wsb_vader script...')

nltk.download('vader_lexicon')

# Define Seaborn style
sns.set(style='darkgrid', context='talk', palette='Dark2')

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

query="SELECT * FROM reddit_comments"
wsb_df = pd.read_sql(query, engine)
wsb_df = wsb_df.drop(['index'], axis=1)
print(wsb_df.head())


# Calculate mean comment score

# We extract the mean of scores:
mean = wsb_df['comment_score'].mean()
print("The average comment score is: {:.2f}".format(mean))

### Create an object of type SentimentIntensityAnalyzer
# creating object of SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

### Calculate Sentiment for All Rows in table

def calculate_sentiment(text):
    # Run VADER on the text
    scores = sid.polarity_scores(text)
    # Extract the compound score
    compound_score = scores['compound']
    return compound_score

def print_sentiment_scores(sentence):
    snt = sid.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(snt)))


# Apply the function to every row in the "Comment" column and output  results into new column "Polarity"
wsb_df['Polarity'] = wsb_df['Comment'].apply(calculate_sentiment)
print(wsb_df['Polarity'].head())

# Sort the data frame and examine the top 10 tweets with the highest compound sentiment.
wsb_df.sort_values(by='Polarity', ascending=False)[:10]

# Sort the data frame and examine the top 10 tweets with the lowest compound sentiment.
wsb_df.sort_values(by='Polarity', ascending=True)[:10]

# create function to compute negative, positive an neutral analysis
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

# Add analysis column to the dataframe
wsb_df['Analysis'] = wsb_df['Polarity'].apply(getAnalysis)

# Update table schema with new columns

colString= """
ALTER TABLE reddit_comments
ADD COLUMN Polarity_vader numeric(10,4),
ADD COLUMN Analysis_vader numeric(10,4);
"""

dbCursor.execute(colString)

# Populate with values
wsb_df.to_sql("reddit_comments", engine, if_exists="replace")

"""
Check distribution and CDF
Source: https://towardsdatascience.com/social-media-sentiment-analysis-with-vader-c29d0c96fa90
In the graph below we can visualize the distribution of compound polarity scores. 
The compound score is a metric that calculates the sum of all lexicon ratings, and normalizes them between -1 (extremely negative) and 1 (extremely positive). 
We can see the that VADER classifies the polarity of sentiment from WSB reddit posts extracted in this sample as mostly neutral
The density value on the Y-axis denotes the scale for 3 possible outcomes (positive, neutra and negative sentiment)
"""

plt.subplot(2,1,1)
plt.title('Distribution Of Sentiments Across all Reddit Posts',fontsize=19,fontweight='bold')
sns.kdeplot(wsb_df['Polarity'],bw_method=0.1)
plt.savefig('test.png')
plt.show()

"""
Plot the CDF of sentiment across Reddit posts
Source: https://www.youtube.com/watch?v=YXLVjCKVP7U
In the graph below we see how sentiment accumulates across the set of WSB posts. Around the central point in the distribution, the gradient of the curve steepens, 
denoting the wider spread of sentiments around the mean (neutral sentiment position.
Defn: The cumulative density function (CDF) of a random variable X is the sum or accrual of probabilities up to some value. 
It shows how the sum of the probabilities approaches 1, which sometimes occurs at a constant rate and sometime occurs at a changing rate. 
The shape of the CF depends on mean and variance."""

plt.subplot(2,1,2)
plt.title('CDF Of Sentiments Across Reddit Posts',fontsize=19,fontweight='bold')
sns.kdeplot(wsb_df['Polarity'],bw_method=0.1,cumulative=True)
plt.xlabel('Sentiment Value',fontsize=19)
plt.show()

"""
### Plot Sentiment Over Time
We can plot how sentiment in the subreddit posts fluctuate over time by first converting the ‘time_created’ column to a
 datetime value and then making it the index of the data frame, which makes it easier to work with time series data."""


# Make date the index of the DataFrame
wsb_df = wsb_df.set_index('time_created')

"""
Then we will group the tweets by month using resample( ), a special method for datetime indices, and calculate the mean( ) 
compound score for each month. Finally, we will plot these averages."""

wsb_df.resample('M')['Polarity'].mean().plot(title="Polarity by Month")

""" Resample by day (‘D’), week (‘W’), or year (‘Y’).
gaps in the graph denote missing points (no comments scraped) in the data series
"""

wsb_df.resample('D')['Polarity'].mean().plot(title="Polarity by Day")

wsb_df.resample('W')['Polarity'].mean().plot(title="Polarity by Week")

"""
Using .loc, we can also zoom in on particular time periods
Trying day before May 13th 2022, and 2 days after that date
"""

wsb_dfMay = wsb_df.loc["2022-05-12":"2022-05-15"].sort_values(by='Polarity')
wsb_dfMay

"""
Check the percentage of positive, neutral and negative tweets in the May data set
"""

# Get the percentage of positive tweets in the data set
print("Sentiment for May:")
pposts = wsb_dfMay[wsb_dfMay.Analysis == 'Positive']
pposts = pposts['Comment']
p = round( (pposts.shape[0] / wsb_dfMay.shape[0]) *100, 1)
print("{}% positive sentiment".format(p))

# Get the percentage of positive tweets in the data set
pposts = wsb_dfMay[wsb_dfMay.Analysis == 'Negative']
pposts = pposts['Comment']
p = round( (pposts.shape[0] / wsb_dfMay.shape[0]) *100, 1)
print("{}% negative sentiment".format(p))

# Get the percentage of positive tweets in the data set
pposts = wsb_dfMay[wsb_dfMay.Analysis == 'Neutral']
pposts = pposts['Comment']
p = round( (pposts.shape[0] / wsb_dfMay.shape[0]) *100, 1)
print("{}% neutral sentiment".format(p))

# Check distribution and CDF of May data

plt.subplot(2,1,1)
plt.title('Distribution Of Sentiments Across Reddit Posts for May',fontsize=19,fontweight='bold')
sns.kdeplot(wsb_dfMay['Polarity'],bw_method=0.1)
plt.show()

# Testing day before Oct 4th 2022, and 2 days after that date

dfOct4 = wsb_df.loc["2022-10-03":"2022-10-06"].sort_values(by='Polarity')
dfOct4.head()

# Check the percentage of positive, neutral and negative tweets in the Oct28 data set

# Get the percentage of positive tweets in the data set
print("Sentiment for Oct 4th:")
pposts = dfOct4[dfOct4.Analysis == 'Positive']
pposts = pposts['Comment']
p = round( (pposts.shape[0] / dfOct4.shape[0]) *100, 1)
print("{}% positive sentiment".format(p))

# Get the percentage of positive tweets in the data set
pposts = dfOct4[dfOct4.Analysis == 'Negative']
pposts = pposts['Comment']
p = round( (pposts.shape[0] / dfOct4.shape[0]) *100, 1)
print("{}% negative sentiment".format(p))

# Get the percentage of positive tweets in the data set
pposts = dfOct4[dfOct4.Analysis == 'Neutral']
pposts = pposts['Comment']
p = round( (pposts.shape[0] / dfOct4.shape[0]) *100, 1)
print("{}% neutral sentiment".format(p))

# Check distribution and CDF of Oct 4th data

plt.subplot(2,1,1)
plt.title('Distribution Of Sentiments Across Reddit Posts 4th Oct',fontsize=19,fontweight='bold')
sns.kdeplot(dfOct4['Polarity'],bw_method=0.1)
plt.show()

# Testing day before Oct 28th 2022, and 2 days after that date

dfOct28 = wsb_df.loc["2022-10-27":"2022-10-31"].sort_values(by='Polarity')
dfOct28.head()

# Create a scatter plot of Oct 28th sentiments by ‘score’ and ‘polarity’

# Plot the polarity
plt.figure(figsize=(8,6))
for i in range (0, dfOct28.shape[0]):
    plt.scatter(dfOct28['Polarity'][i], dfOct28['comment_score'][i], color='Blue')

plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Score')
plt.show()

# Check the percentage of positive, neutral and negative comments in the Oct28 data set

# Get the percentage of positive comments in the data set
print("Sentiment for Oct 28th:")
pposts = dfOct28[dfOct28.Analysis == 'Positive']
pposts = pposts['Comment']
p = round( (pposts.shape[0] / dfOct28.shape[0]) *100, 1)
print("{}% positive sentiment".format(p))

# Get the percentage of negative comments in the data set
nposts = dfOct28[dfOct28.Analysis == 'Negative']
nposts = nposts['Comment']
n = round( (nposts.shape[0] / dfOct28.shape[0]) *100, 1)
print("{}% negative sentiment".format(n))

# Get the percentage of negative comments in the data set
zposts = dfOct28[dfOct28.Analysis == 'Neutral']
zposts = zposts['Comment']
n = round( (zposts.shape[0] / dfOct28.shape[0]) *100, 1)
print("{}% neutral sentiment".format(n))

# Create a FULL scatter plot of sentiments by ‘score’ and ‘polarity’

# Plot the polarity
plt.figure(figsize=(8,6))
for i in range (0, wsb_df.shape[0]):
    plt.scatter(wsb_df['Polarity'][i], wsb_df['comment_score'][i], color='Blue')

plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Score')
plt.show()

# Get the percentage of negative tweets in the data set
nposts = wsb_df[wsb_df.Analysis == 'Negative']
nposts = nposts['Comment']
n = round( (nposts.shape[0] / wsb_df.shape[0]) *100, 1)
print("{}% negative sentiment".format(n))

# Get the percentage of neutral tweets in the data set
zposts = wsb_df[wsb_df.Analysis == 'Neutral']
zposts = zposts['Comment']
z = round( (zposts.shape[0] / wsb_df.shape[0]) *100, 1)
print("{}% neutral sentiment".format(z))


# Create a bar chart of sentiments

# show the value counts
wsb_df['Analysis'].value_counts()

# plot and visualise the counts
plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Counts')
wsb_df['Analysis'].value_counts().plot(kind='bar')
plt.show()

print("End of sentiment_wsb_vader script.")