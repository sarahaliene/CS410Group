import nltk
import os
import pandas as pd
from nltk.corpus import twitter_samples
from nltk.twitter import json2csv
from nltk.tokenize import TweetTokenizer
nltk.download("twitter_samples")

# Ensure the 'data' folder exists
os.makedirs("data", exist_ok=True)

files_list = twitter_samples.fileids()

# Double check that printed file names are same as saved ones
print(files_list)

pos_tweet_file = twitter_samples.abspath("positive_tweets.json")
neg_tweet_file = twitter_samples.abspath("negative_tweets.json")
tweet_file = twitter_samples.abspath("tweets.20150430-223406.json")

with open(pos_tweet_file) as fp:
	json2csv(fp, "data/pos_tweets_text.csv", ['text'])

with open(neg_tweet_file) as fp:
	json2csv(fp, "data/neg_tweets_text.csv", ['text'])

with open(tweet_file) as fp:
	json2csv(fp, "data/tweets_text.csv", ['text'])

