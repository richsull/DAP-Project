## Elon Musk Tweets - Sentiment Analysis

## Sentiment Analysis of Elon Musk tweets Jan 2022 - Oct 2022


## Data set sourced through a Twitter API scrape. Python code source: https://www.youtube.com/watch?v=ujId4ipkBio&t=171s

# Sentiment analysis is form of text analytics that uses natural language processing (NLP) and machine learning.
# It is sometimes referred to as 'opinion mining'. A key aspect of sentiment analysis is polarity classification.
# Polarity refers to the overall sentiment conveyed by a particular tweet, phrase or word.
# Polarity can be expressed as a simple numerical score (or rating).
# In this example, the score is represneted as 'below 0', 'zero', or 'above zero', with zero representing a neutral sentiment.

# Import libraries
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
# import demoji
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
import psycopg2
from sqlalchemy import create_engine
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
print("Start of SentAlys_MuskTweets script...")


# Import data from PostgreSQL database and create a data frame to store the data

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

query="SELECT * FROM tweets "
tweets_df = pd.read_sql(query, engine)
# print(tweets_df.head())
print(len(tweets_df))

# Create two separate functions to check the subjectivity and polarity of the text in a given tweet.
# Subjectivity is our proxy for 'opinionated' text, while polarity is a simple measure of the positivity or negativty of the tweet.

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

# Next, create two columns in the data frame to store the subjectivity and polarity data

tweets_df['Subjectivity'] = tweets_df["text"].apply(getSubjectivity)
tweets_df['Polarity'] = tweets_df["text"].apply(getPolarity)
tweets_df

# Plot a wordcloud to see how well the sentiments are distributed.
# The cloud image will also provide a simple visualisation of  words and terms that are common across tweets

allWords = ' '.join(tweets_df.iloc[:,4])
wordCloud = WordCloud(width=500, height=300, random_state = 21, max_font_size =119).generate(allWords)

plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Create a function to compute simple negative, neutral and positive scores for the sentiment analysis.
# Scores are based on a simple median of zero.
# String values ar retuned on the basis of the score value. Last, create a new column called 'Analysis'

# create function to compute negative, positive an neutral analysis
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

# define the new dataframe
tweets_df['Analysis'] = tweets_df['Polarity'].apply(getAnalysis)
#tweets_df

# Split the data frame to show only the relevant columns
dfNew = tweets_df[['text','Subjectivity','Polarity', 'Analysis']]



## Create a scatter plot of sentiments by 'subjectivity' and 'polarity'

# Plot the polarity and subjectvitiy
plt.figure(figsize=(8,6))
for i in range (0, tweets_df.shape[0]):
    plt.scatter(dfNew['Polarity'][i], dfNew['Subjectivity'][i], color='Blue')

plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()

# Check the percentage of positive, neutral and negative tweets in the data set

# Get the percentage of positive tweets in the data set
ptweets = tweets_df[tweets_df.Analysis == 'Positive']
ptweets = ptweets['text']
pos = round( (ptweets.shape[0] / tweets_df.shape[0]) *100, 1)
print("{}% positive sentiment".format(pos))

# Get the percentage of negative tweets in the data set
ntweets = tweets_df[tweets_df.Analysis == 'Negative']
ntweets = ntweets['text']
neg = round( (ntweets.shape[0] / tweets_df.shape[0]) *100, 1)
print("{}% negative sentiment".format(neg))

# Get the percentage of neutral tweets in the data set
ntweets = tweets_df[tweets_df.Analysis == 'Neutral']
ntweets = ntweets['text']
ntr = round( (ntweets.shape[0] / tweets_df.shape[0]) *100, 1)
print("{}% neutral sentiment".format(ntr))

# Create a bar chart of tweet sentiments

# show the value counts
tweets_df['Analysis'].value_counts()

# plot and visualise the counts
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
tweets_df['Analysis'].value_counts().plot(kind='bar')
plt.show()

# plot sentiment over time

#ax = plt.gca()
#plt.title('Polarity of comments over time')
#plt.xlabel('Date')
#plt.ylabel('Polarity')
#tweets_df.plot(x = 'datetime', y = 'Polarity', ax=ax)
ax = plt.gca()
plt.title('Polarity of comments over time')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.ylabel('Polarity')
tweets_df.plot(x = 'datetime', y = 'Polarity', ax=ax)
plt.gcf().autofmt_xdate()
plt.show()

print("End of SentAlys_MuskTweets script.")