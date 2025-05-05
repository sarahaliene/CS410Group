import os
import pandas as pd
from collections import Counter
from predicttweet import predict_tweet


def test_predictions(tweets_file_name):
    tweets = pd.read_csv(tweets_file_name)

    tweets_list = []
    prediction_list = []
    prediction_counter = Counter()

    #Add each line (tweet) to list
    for index, row in tweets.iloc[1:].iterrows():

        tweets_list.append(row)

    #Get prediction for each tweet in list
    for tweet in tweets_list:
        pred = predict_tweet(tweet)
        prediction_list.append(pred)

    prediction_counter.update(prediction_list)
    print("Positive:", prediction_counter['positive'])
    print("Negative:", prediction_counter['negative'])
    print(prediction_list[:50])