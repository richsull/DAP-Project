## WallStreetBets - Reddit Sentiment Analysis

# Sentiment Analysis of posts on r/wallstreetbets subreddit:  Dec 2022 - Dec 2022

# Data set sourced through a Twitter API scrape and stored in PostreSQL. Python code source: https://www.youtube.com/watch?v=ujId4ipkBio&t=171s

# Sentiment analysis is form of text analytics that uses natural language processing (NLP) and machine learning. It is sometimes
# referred to as 'opinion mining'. A key aspect of sentiment analysis is polarity classification. Polarity refers to the overall
#  sentiment conveyed by a particular tweet, phrase or word. Polarity can be expressed as a simple numerical score (or rating).
# In this example, the score is represented as 'below 0', 'zero', or 'above zero', with zero representing a neutral sentiment.

# import the packages
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

print("Start of SentAlys_WSB script...")

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

query="SELECT * FROM reddit_comments"
wsb_df = pd.read_sql(query, engine)
wsb_df = wsb_df.drop(['index'], axis=1)
print(wsb_df.head())

# Create two separate functions to check the subjectivity and polarity of the text in a given reddit post.
# Subjectivity is our proxy for 'opinionated' text, while polarity is a simple measure of the positivity or negativty of the tweet.

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

# Next, create two columns in the data frame to store the subjectivity and polarity data

wsb_df['Subjectivity'] = wsb_df["Comment"].apply(getSubjectivity)
wsb_df['Polarity'] = wsb_df["Comment"].apply(getPolarity)
wsb_df

# Plot a wordcloud to see how well the sentiments are distributed.
# The cloud image will also provide a simple visualisation of words and terms that are common across tweets

allWords = ' '.join(wsb_df['Comment'])
wordCloud = WordCloud(width=500, height=300, random_state = 30, max_font_size =119).generate(allWords)

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

# show the new dataframe
wsb_df['Analysis'] = wsb_df['Polarity'].apply(getAnalysis)
wsb_df

# Split the data frame to show only the relevant columns

dfNew = wsb_df[['Comment','Subjectivity','Polarity', 'Analysis']]
dfNew.head(10)

# Create a scatter plot of sentiments by ‘subjectivity’ and ‘polarity’

# Plot the polarity and subjectvitiy
plt.figure(figsize=(8,6))
for i in range (0, dfNew.shape[0]):
    plt.scatter(dfNew['Polarity'][i], dfNew['Subjectivity'][i], color='Blue')

plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()

# Check the percentage of positive, neutral and negative tweets in the data set

# Get the percentage of positive tweets in the data set
pposts = dfNew[dfNew.Analysis == 'Positive']
pposts = pposts['Comment']
p = round( (pposts.shape[0] / dfNew.shape[0]) *100, 1)
print("{}% positive sentiment".format(p))

# Get the percentage of negative tweets in the data set
nposts = dfNew[dfNew.Analysis == 'Negative']
nposts = nposts['Comment']
n = round( (nposts.shape[0] / dfNew.shape[0]) *100, 1)
print("{}% negative sentiment".format(n))

# Get the percentage of neutral tweets in the data set
zposts = dfNew[dfNew.Analysis == 'Neutral']
zposts = zposts['Comment']
z = round( (zposts.shape[0] / dfNew.shape[0]) *100, 1)
print("{}% neutral sentiment".format(z))

# show the value counts
dfNew['Analysis'].value_counts()

# plot and visualise the counts
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
dfNew['Analysis'].value_counts().plot(kind='bar')

plt.show()

# Plot polarity over time

ax = plt.gca()
plt.title('Polarity of comments over time')
plt.xlabel('Date')
plt.ylabel('Polarity')
wsb_df.plot(x = 'time_created', y = 'Polarity', ax=ax)
plt.gcf().autofmt_xdate()
plt.show()


print("End of SentAlys_WSB script.")