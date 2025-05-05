import nltk
import os
import pandas as pd
import re
from vocab import create_vocab
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.twitter import json2csv
from nltk.tokenize import TweetTokenizer
nltk.download("twitter_samples")
nltk.download('stopwords')

#Functions as of 5-5-25

#Hyperlinks
#hyperlinks = r"https?:\/\/[^\s]+[\r\n]*"

#Stopwords
stop_words = set(stopwords.words("english"))

#Set up url words to remove
stop_words.add("www")
stop_words.add("http")
stop_words.add("https")
stop_words.add("href")
stop_words.add("ahref")

#Set up TweetTokenizer
tokenizer = TweetTokenizer(strip_handles = True, reduce_len = True)

#Function for hyperlink removal
#def remove_hyperlinks(text):

    #if isinstance(text, str):
        #return re.sub(hyperlinks, '', text)
    #return text

#Function for stopword removal
def remove_stopwords(tweet):
    return [word for word in tweet if word not in stop_words]

#Function to tokenize the tweets
def tweet_tokenizer(tweet):
    return tokenizer.tokenize(tweet)

# Ensure the 'data' folder exists
os.makedirs("data", exist_ok=True)

#Get the twitter data
files_list = twitter_samples.fileids()

#Double check that printed file names are same as saved ones
#print(files_list)

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

#Pos_dataset
#pos_tweet_list = []
#for index, row in pos_dataset.iterrows():
    #line = row[0]
    #text_chunk = tokenizer.tokenize(line)

    #new_text_chunk = []
    
#Remove tokens with "http"
    #for token in text_chunk:

        #if "http" not in token:
            #new_text_chunk.append(token)
        #else:
            #continue
    #pos_tweet_list.append(new_text_chunk)


#Neg_dataset
#neg_tweet_list = []
#for index, row in neg_dataset.iterrows():
    #line = row[0]
    #text_chunk = tokenizer.tokenize(line)

    #new_text_chunk = []
    
#Remove tokens with "http"
    #for token in text_chunk:

        #if "http" not in token:
            #new_text_chunk.append(token)
        #else:
            #continue
    #neg_tweet_list.append(new_text_chunk)


#Use pandas to read the csv files
#pos_dataset = pd.read_csv("data/pos_tweets_text.csv")
#neg_dataset = pd.read_csv("data/neg_tweets_text.csv")
#tweets = pd.read_csv("data/tweets_text.csv")

#Apply the remove_hyperlinks function to the tweets
#pos_dataset = pos_dataset.applymap(remove_hyperlinks)
#neg_dataset = neg_dataset.applymap(remove_hyperlinks)
#tweets = tweets.applymap(remove_hyperlinks)

#Put the results back into the csv files
#pos_dataset.to_csv('data/pos_tweets_text.csv', index=False)
#neg_dataset.to_csv('data/neg_tweets_text.csv', index=False)
#tweets.to_csv('data/tweets_text.csv', index=False)

#Read the data (again....)
pos_tweets = pd.read_csv("data/pos_tweets_text.csv")
neg_tweets = pd.read_csv("data/neg_tweets_text.csv")
tweets = pd.read_csv("data/tweets_text.csv")

#Apply the tweet_tokenizer function to the tweets
pos_tweets["text"] = pos_tweets["text"].apply(tweet_tokenizer)
neg_tweets["text"] = neg_tweets["text"].apply(tweet_tokenizer)
tweets["text"] = tweets["text"].apply(tweet_tokenizer)

#Apply the remove_stopwords function to the tweets
pos_tweets["text"] = pos_tweets["text"].apply(remove_stopwords)
neg_tweets["text"] = neg_tweets["text"].apply(remove_stopwords)
tweets["text"] = tweets["text"].apply(remove_stopwords)

# keep only alnum words
# def keep_alnum(tweet):
#   return [word for word in tweet if word.isalnum()]
# pos_tweets["text"] = pos_tweets["text"].apply(keep_alnum)
# neg_tweets["text"] = neg_tweets["text"].apply(keep_alnum)
# tweets["text"] = tweets["text"].apply(keep_alnum)

#Save the processed data in a new folder
os.makedirs("processed_data", exist_ok=True)
pos_tweets.to_csv("processed_data/pos_tweets.csv", index=False)
neg_tweets.to_csv("processed_data/neg_tweets.csv", index=False)
tweets.to_csv("processed_data/tweets.csv", index=False)

#Vocab - takes n as parameter
#top_pos = create_vocab(pos_tweets)
#top_neg = create_vocab(neg_tweets)

print("Finished processing")
