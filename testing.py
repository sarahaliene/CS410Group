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
    prediction_printable = []
    vader_list = []
    prediction_counter = Counter()

    #Get prediction for each tweet in list
    for tweet in tweets_list:

        pred = predict_tweet(tweet)

        vader_dictionary = get_vader_sentiment(tweet)
        vader_score = vader_dictionary["compound"]

        prediction_pair = (pred, vader_score)
        prediction_printable.append(prediction_pair)

        prediction_list.append(pred)
        vader_list.append(vader_score)

    prediction_counter.update(prediction_list)
    print("Positive:", prediction_counter['positive'])
    print("Negative:", prediction_counter['negative'])
    print(prediction_printable[:25])

    return(prediction_printable)

def compare_vader_bayes(predictions):
    #Positive > 0.05
    #Negative < - 0.05
    #Neutral between those two
    count = 0

    printable_list = []
    pred_counter = Counter()

    for pair in predictions:
        pred, vader = pair

        #If the two agree
        if (pred == "positive") and (vader > 0.05):
            prediction = ("Bayes positive, Vader positive")

        elif (pred == "negative") and (vader < -0.05):
            prediction = ("Bayes negative, Vader negative")

        #If the two are opposite
        elif (pred == "positive") and (vader < -0.05):
            prediction = ("Bayes positive, Vader negative")

        elif (pred == "negative") and (vader > 0.05):
            prediction = ("Bayes negative, Vader positive")

        #If vader is neutral
        elif (pred == "positive") and (-0.05 < vader < 0.05):
            prediction = ("Bayes positive, Vader neutral")

        elif (pred == "negative") and (-0.05 < vader < 0.05):
            prediction = ("Bayes negative, Vader neutral")

        #Add to list
        printable_list.append(prediction)

    #Update counter
    pred_counter.update(printable_list)
    print(pred_counter)
