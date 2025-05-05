from collections import Counter
import pandas as pd
import os
import pickle

def create_vocab(tweet_file_name):

    tweet_counter = Counter()

    tweets_file = open(tweet_file_name, 'rb')    
    tweets_list = pickle.load(tweets_file)

    for tweet in tweets_list:

        tweet_counter.update(tweet)


    top_200 = tweet_counter.most_common(200)

    print("Finished vocab creation")
    return top_200
