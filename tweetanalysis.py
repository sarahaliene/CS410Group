import os
from naivebayestweets import naive_bayes_tweet
from vocab import create_vocab
from predicttweet import predict_tweet
from testing import test_predictions

#Run entire processing script
with open("processing.py") as file:
    exec(file.read())

#Set path for accessing processed csv files
current_path = os.path.dirname(__file__)
pos_tweets = current_path + "/processed_data/pos_tweets.csv"
neg_tweets = current_path + "/processed_data/neg_tweets.csv"
test_tweets = current_path + "/processed_data/tweets.csv"

#Initialize naive_bayes values
naive_bayes_tweet(pos_tweets, neg_tweets, 1.0)

#Get top 200 words for pos and neg tweets
top_200_pos = create_vocab(pos_tweets)
top_200_neg = create_vocab(neg_tweets)

#Print top 10 of each
print("Top 10 Positive Words:", top_200_pos[:10])
print("Top 10 Negative Words:", top_200_neg[:10])

#Print predictions for pre-labeled files
print("Testing pre-labeled positive tweets using Naive Bayes")
test_predictions(pos_tweets)

print("Testing pre-labeled negative tweets using Naive Bayes")
test_predictions(neg_tweets)

print("Predictions for unlabeled test file tweets using Naive Bayes")
test_predictions(test_tweets)
