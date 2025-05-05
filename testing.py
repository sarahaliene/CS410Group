import os
import pandas as pd
import pickle
from collections import Counter
from predicttweet import predict_tweet
from vadersentiment import get_vader_sentiment


def test_predictions(tweets_file_name):

    tweets_file = open(tweets_file_name, 'rb')    
    tweets_list = pickle.load(tweets_file)


    prediction_list = []
    vader_list = []
    prediction_counter = Counter()

    #Get prediction for each tweet in list
    for tweet in tweets_list:

        pred = predict_tweet(tweet)
        vader_score = get_vader_sentiment(tweet)
        prediction_list.append(pred)
        vader_list.append(vader_score)

    prediction_counter.update(prediction_list)
    print("Positive:", prediction_counter['positive'])
    print("Negative:", prediction_counter['negative'])
    print(prediction_list[:50], vader_list[:50])
