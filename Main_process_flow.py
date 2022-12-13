## Main process flow script for project

# Source code: https://stackoverflow.com/questions/64016426/how-to-run-multiple-python-scripts-using-single-python-py-script


import glob,os
import datetime


os.chdir(r"C:\Users\mark1\PycharmProjects\pythonProject1")  # locate ourselves in the directory
print("Begin workflow.")
print("Sit back and have a cup of tea, this process can take approximately 40 minutes to run")

start = datetime.datetime.now()


for script in ["stock_database_0_92.py", "social_to_mongo.py","mongo_to_postgres.py", "SentAlys_WSB.py", "SentAlys_MuskTweets.py", "sentiment_wsb_vader.py", "sentiment_tweet_vader.py"]:

    with open(script) as f:
       contents = f.read()
    exec(contents)

end = datetime.datetime.now()

time_elapsed = end-start
print("End of workflow.")
print("Time elapsed = ",time_elapsed)