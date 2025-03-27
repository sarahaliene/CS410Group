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

tokenizer = TweetTokenizer(strip_handles = True, reduce_len = True)

pos_dataset = pd.read_csv("pos_tweets_text.csv")
neg_dataset = pd.read_csv("neg_tweets_text.csv")

for index, row in pos_dataset.iterrows():
	line = row[0]
	text_chunk = tokenizer.tokenize(line)

	new_text_chunk = []
	

	for token in text_chunk:

		if "http" not in token:
			new_text_chunk.append(token)
		else:
			continue

