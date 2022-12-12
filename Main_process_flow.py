## Main process flow script for project

# Source code: https://stackoverflow.com/questions/64016426/how-to-run-multiple-python-scripts-using-single-python-py-script


import glob,os
os.chdir(r"C:\Users\mark1\PycharmProjects\pythonProject1")  # locate ourselves in the directory
print("Begin workflow.")

for script in ["stock_database_0_92.py", "social_to_mongo.py","mongo_to_postgres.py", "SentAlys_WSB.py", "SentAlys_MuskTweets.py"]:

    with open(script) as f:
       contents = f.read()
    exec(contents)

print("End of workflow.")