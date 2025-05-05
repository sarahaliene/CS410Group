from collections import Counter
import pandas as pd
import os
#Functioning as of 5-5-25

def create_vocab(tweet_file_name):

    tweet_counter = Counter()

    tweets = pd.read_csv(tweet_file_name)
    tweets_list = []

    #Add each line (tweet) to the counter
    for index, row in tweets.iloc[1:].iterrows():

        tweets_list.append(row)

    for tweet in tweets_list:

        tweet_counter.update(tweet)


    top_200 = tweet_counter.most_common(200)

    print("Finished vocab creation")
    return top_200
