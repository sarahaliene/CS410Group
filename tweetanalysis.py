import os
from naivebayestweets import naive_bayes_tweet

#Run entire processing script
with open("processing.py") as file:
    exec(file.read())

current_path = os.path.dirname(__file__)

pos_tweets = current_path + "/processed_data/pos_tweets.csv"
neg_tweets = current_path + "/processed_data/neg_tweets.csv"
test_tweets = current_path + "/processed_data/tweets.csv"

naive_bayes_tweet(pos_tweets, neg_tweets, 1.0)
