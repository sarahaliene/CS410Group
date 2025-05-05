import nltk
import os
import pandas as pd
import re
import pickle
from vocab import create_vocab
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.twitter import json2csv
from nltk.tokenize import TweetTokenizer
nltk.download("twitter_samples")
nltk.download('stopwords')

#Stopwords
stop_words = set(stopwords.words("english"))

#Set up TweetTokenizer
tokenizer = TweetTokenizer(strip_handles = True, reduce_len = True)

# Ensure the 'data' folder exists
os.makedirs("data", exist_ok=True)

#Get the twitter data
files_list = twitter_samples.fileids()

#Link to downloaded json files
pos_tweet_file = twitter_samples.abspath("positive_tweets.json")
neg_tweet_file = twitter_samples.abspath("negative_tweets.json")
tweet_file = twitter_samples.abspath("tweets.20150430-223406.json")

#Convert json files to csv
with open(pos_tweet_file) as fp:
    json2csv(fp, "data/pos_tweets_text.csv", ['text'])

with open(neg_tweet_file) as fp:
    json2csv(fp, "data/neg_tweets_text.csv", ['text'])

with open(tweet_file) as fp:
    json2csv(fp, "data/tweets_text.csv", ['text'])

pos_dataset = pd.read_csv("data/pos_tweets_text.csv")
neg_dataset = pd.read_csv("data/neg_tweets_text.csv")
test_dataset = pd.read_csv("data/tweets_text.csv")


#Pos_dataset
pos_tweet_list = []
for index, row in pos_dataset.iterrows():
    line = row[0]
    text_chunk = tokenizer.tokenize(line)

    new_text_chunk = []
    
#Remove tokens with "http"
    for token in text_chunk:

        if ("http" not in token) and (token not in stop_words):
            new_text_chunk.append(token)
        else:
            continue
    
    pos_tweet_list.append(new_text_chunk)


#Neg_dataset
neg_tweet_list = []
for index, row in neg_dataset.iterrows():
    line = row[0]
    text_chunk = tokenizer.tokenize(line)

    new_text_chunk = []
    
#Remove tokens with "http"
    for token in text_chunk:

        if ("http" not in token) and (token not in stop_words):
            new_text_chunk.append(token)
        else:
            continue

    neg_tweet_list.append(new_text_chunk)

#Test dataset
test_tweet_list = []
for index, row in test_dataset.iterrows():
    line = row[0]
    text_chunk = tokenizer.tokenize(line)

    new_text_chunk = []
    
#Remove tokens with "http"
    for token in text_chunk:

        if ("http" not in token) and (token not in stop_words):
            new_text_chunk.append(token)
        else:
            continue
            
    test_tweet_list.append(new_text_chunk)

#Save the processed data in a new folder
os.makedirs("processed_data", exist_ok=True)

#Pickle the lists to access later
with open('pos_tweets.pkl', 'wb') as fp:
    pickle.dump(pos_tweet_list, fp)

with open('neg_tweets.pkl', 'wb') as fp:
    pickle.dump(neg_tweet_list, fp)

with open('tweets.pkl', 'wb') as fp:
    pickle.dump(test_tweet_list, fp)

print("Finished processing")
