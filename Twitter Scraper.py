# Twitter scraper

# Source: https://medium.com/machine-learning-mastery/how-to-scrape-millions-of-tweets-using-snscraper-aa47cee400ec
import snscrape.modules.twitter as sntwitter
import pandas as pd

# Setting variables to be used below
maxTweets = 10000
# Creating list to append tweet data to
tweets_list2 = []
# Using TwitterSearchScraper to scrape data and append tweets to list
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:elonmusk (twitter OR twtr OR tsla OR tesla) since:2022-01-01 until:2022-10-28').get_items()):
    if i>maxTweets:
        break
    tweets_list2.append([tweet.date, tweet.id, tweet.rawContent, tweet.user.username, tweet.quoteCount, tweet.likeCount])


# Creating a dataframe from the tweets list above
tweets_df2 = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet Id', 'Text', 'Username', 'LikeCount', 'QuoteCount'])

tweets_df2.to_csv("elonmusk.csv", index = False)