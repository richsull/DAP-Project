# Using Python, connect to MongoDB, process databases, extract all databases, upload to PostgreSQL on Google Cloud Platform

# Source for writing dataframes to SQL: https://stackoverflow.com/questions/23103962/how-to-write-dataframe-to-postgres-table



# pip install pymongo
from pymongo import MongoClient
import pandas as pd
import psycopg2
import re
from sqlalchemy import create_engine

print("Start of mongo_to_postgres script...")

# Connect to MongoDB Atlas

cluster = "mongodb+srv://markdata:DAPproject@dapcluster.hbqohv5.mongodb.net/?retryWrites=true&w=majority"
# user = 'markdata'
# password = 'DAPproject'
client =  MongoClient(cluster)

# Check database names
print(client.list_database_names())

# Check collection names
db = client["DAPproject-twitter"]
print(db.list_collection_names())

# Extract Reddit threads
reddit_threads = []
collection=db['reddit_threads']
cursor = collection.find({})
thread_list = list(cursor)
threads = pd.DataFrame(thread_list)

# Drop Unnecessary _id column
threads = threads.drop(['_id'], axis=1)

# Function to clean text

def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+_', '', text) # remove @mentions based on any character (raw string)
    text = re.sub(r'#', '', text) # remove the '#' symbol and replace with empty string
    text = re.sub('\n', '', text) # remove linebreaks and replace with space
    text = re.sub('RT[\s]+', '', text)  # remove retweets (RT) followed by white space
    text = re.sub('https?:\/\/\S+', '', text) # remove URL hyperlinks followed by one or more white spaces
    return text

# apply the function
threads["body"] = threads['body'].apply(cleanTxt)

# Extract Reddit comments

redditcomments = []
comcollection=db['reddit_comments']
comcursor = comcollection.find({})
comment_list = list(comcursor)
comments = pd.DataFrame(comment_list)

# Drop _id column created in MondoDB
comments = comments.drop(['_id'], axis=1)

# Drop first 3 characters of parent_id to make them the same as the thread id
comments['parent_id'] = comments['parent_id'].str[3:]
# Function to clean text

# apply the function
comments["Comment"] = comments['Comment'].apply(cleanTxt)

# Extract Tweets

tweets = []
twcollection=db['tweets']
twcursor = twcollection.find({})
tweet_list = list(twcursor)
tweets = pd.DataFrame(tweet_list)
# Drop id column
tweets = tweets.drop(['_id'], axis=1)
# apply the function
tweets["text"] = tweets['text'].apply(cleanTxt)

# Upload to PostgreSQL

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

## Create tables

# Create reddit threads table
threadString = """
CREATE TABLE IF NOT EXISTS reddit_threads(
ind numeric(15,4),
title VARCHAR(200),
date_time TIMESTAMP,
body VARCHAR(200),
id VARCHAR(200) PRIMARY KEY,
score numeric(15,4),
comments numeric(15,4)
);
"""
# Create Reddit comments table
commentstring = """
CREATE TABLE IF NOT EXISTS reddit_comments(

comment VARCHAR(10000),
author VARCHAR(200),
comment_score numeric(15,4),
parent_id VARCHAR(200) PRIMARY KEY,
date_time TIMESTAMP
);
"""
# Create twitter threads table
twitterString = """
CREATE TABLE IF NOT EXISTS tweets(
tweet_id numeric(20,4) PRIMARY KEY,
date_time TIMESTAMP,
text VARCHAR(200),
username VARCHAR(200),
quotecount numeric(15,4),
likecount numeric(15,4),
replycount numeric(15,4),
retweetcount numeric(15,4),
tweetinreply numeric(15,4)
);
"""
# Execute create table commands
dbCursor.execute(threadString)
dbCursor.execute(commentstring)
dbCursor.execute(twitterString)

# define engine
engine = create_engine(f'postgresql://{username}:{password1}@{host1}:5432/{database1}')
# Populate tables
tweets.to_sql("tweets", engine, if_exists='replace')
threads.to_sql("reddit_threads", engine, if_exists='replace')
comments.to_sql("reddit_comments", engine, if_exists='replace')

print("End of mongo_to_postgres script.")