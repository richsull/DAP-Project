### Social media extraction to MongoDB upload
### Extraction from Reddit and Twitter
### Source for Reddit: https://www.geeksforgeeks.org/scraping-reddit-using-python/
### Source for Twitter:
### Modified by Mark
### Note: takes approx 20 minutes to run the Reddit comment extraction section

# Import packages
import requests
import praw
import pandas as pd
from pymongo import MongoClient
import snscrape.modules.twitter as sntwitter
import json

print("Start of social_to_mongo script...")

# Connect to Reddit API using PRAW

reddit1 = praw.Reddit(
    client_id="OvH5mhoPWl0DvkYPWaG9wg",
    client_secret="Hl87lQTiwTAigHPjs6QJeaumLRfOuw",
    password="NJdVr7hDXsAfgwEbUnfs",
    user_agent="DAP_scrape2",
    username="McNData")
subred = "WallStreetBets"
subreddit = reddit1.subreddit(subred)

# Extract threads from r/wallstreetbetes
posts = reddit1.subreddit("wallstreetbets")
posts_data = []
for post in posts.search(("musk"), limit=500):
    posts_data.append([post.title,int(post.created), post.selftext,post.id,post.score,post.num_comments])

# Convert Epoch time to datetime
reddit_df = pd.DataFrame(posts_data, columns = ['title','created','body', 'id','score','num_comments'])
# Filter for posts only with Twitter in title
reddit_df = reddit_df[reddit_df["title"].str.contains("Twitter") == True]
# Create dataframe for threads
reddit_df['created'] = pd.to_datetime(reddit_df['created'], unit='s')
# reset the index
reddit_df = reddit_df.reset_index(drop=True)

# Upload threads to MongoDB Atlas
# Connect to MongoDB Atlas

cluster = "mongodb+srv://markdata:DAPproject@dapcluster.hbqohv5.mongodb.net/?retryWrites=true&w=majority"
# user = 'markdata'
# password = 'DAPproject'
client =  MongoClient(cluster)

# Check database names
#print(client.list_database_names())
# Connect to database
db = client["DAPproject-twitter"]
#print(db.list_collection_names())

# Drop threads collection and recreate to avoid repeats
db.reddit_threads.drop()
# Convert Reddit threads dataframe to dictionary and send to MongoDB collection
db.reddit_threads.insert_many(reddit_df.apply(lambda x: x.to_dict(), axis=1).to_list())
# Check number of documents
db.reddit_threads.count_documents({})

# Iterate over threads collection to IDs and then extract all comments
thread_id = db.reddit_threads.find().distinct('id')
# Extract all comments
print('Beginning Reddit comment extraction. Be patient, this can take over 20 minutes.')
comments_df = []
for n, name in enumerate(thread_id):
    submission = reddit1.submission(id=thread_id[n])
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        comments_df.append([comment.parent_id, comment.body, comment.author, comment.score, comment.created_utc])
# Convert list to DataFrame
commentsdf = pd.DataFrame(comments_df, columns=['parent_id','Comment','author', 'comment_score', 'time_created'])
# Convert from epoch to New York time to align with stock price times
commentsdf['time_created'] = pd.to_datetime(commentsdf['time_created'], unit='s')
# Reset index
commentsdf = commentsdf.drop(index=0)
commentsdf.index = range(len(commentsdf.index))
# Author column giving errors when converting so dropping
commentsdf = commentsdf.drop(['author'], axis=1)
# Check length of commentsdf
print(len(commentsdf))
print('End of Reddit comment extraction.')

# Drop comments collection and recreate to avoid repeated data
db.reddit_comments.drop()
# Convert Reddit comments dataframe to dictionary and send to MongoDB collection
db.reddit_comments.insert_many(commentsdf.apply(lambda x: x.to_dict(), axis=1).to_list())

## Extract Twitter data


# Set max number of tweets
maxTweets = 10000
# Creating list to append tweet data to
tweets_list2 = []
# Using TwitterSearchScraper to scrape data and append tweets to list
users = ["elonmusk"]
for j, user in enumerate(users):
    # Choose date range to search
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:{} since:2021-12-01'.format(users[j])).get_items()):
        if i>maxTweets:
            break
        tweets_list2.append([ tweet.id, tweet.date, tweet.rawContent, tweet.user.username, tweet.quoteCount, tweet.likeCount, tweet.replyCount, tweet.retweetCount, tweet.inReplyToTweetId, tweet.mentionedUsers])

# Creating a dataframe from the tweets list above
tweets_df2 = pd.DataFrame(tweets_list2, columns=['tweet_id','datetime', 'text', 'username', 'quotecount', 'likecount', 'replycount', 'retweetcount', 'tweetinreply', 'mentionedusers'])

# Convert epoch to New York time to align with stock price times
tweets_df2['datetime'] = pd.to_datetime(tweets_df2['datetime'], unit='s').dt.tz_convert('America/New_York')
tweets_df2 = tweets_df2.drop(['mentionedusers'], axis=1)
# Save tweets as JSON and upload to MongoDB
tweets_df2.reset_index().to_json('twitter.json',orient='records', date_format='iso', indent=4)
with open('twitter.json') as file:
    file_data = json.load(file)

# Drop comments collection and recreate to avoice repeated data
db.tweets.drop()
# Attach JSON to collection
db.tweets.insert_many(file_data)


print("End of social_media_to_mongo script.")