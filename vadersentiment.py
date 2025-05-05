import nltk
import os
import pandas as pd

from nltk import tokenize

#NLTK Sentiment Analyzer tool
from nltk.sentiment import sentiment_analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def get_vader_sentiment(tweet):
    sia = SentimentIntensityAnalyzer()
    tweet_string = ""

    for word in tweet:
        tweet_string += " " + word

    tweet_score = sia.polarity_scores(tweet_string)

    return(tweet_score)

#print(get_vader_sentiment("RT @LabourEoin: The economy was growing 3 times faster on the day David Cameron became Prime Minister than it is today.. #BBCqt http://t.coâ€¦"))