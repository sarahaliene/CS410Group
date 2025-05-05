import os
import pickle
from collections import Counter


def naive_bayes_tweet(pos_tweets_file_name, neg_tweets_file_name, laplace):

  pos_tweets = pd.read_csv(pos_tweets_file_name)
  neg_tweets = pd.read_csv(neg_tweets_file_name)

  #Lists for holding tweets
  pos_tweets_list = []
  neg_tweets_list = []

  for index, row in pos_tweets.iterrows():
    pos_tweets_list.append(row)

  for index, row in neg_tweets.iterrows():
    neg_tweets_list.append(row)

  #Get the number of pos & neg tweets
  total_pos_tweets = len(pos_tweets_list)
  total_neg_tweets = len(neg_tweets_list)

  #Lists for pos and neg words
  pos_word_list = []
  neg_word_list = []

  #Sets for unique words
  pos_word_set = set()
  neg_word_set = set()
  total_word_set = set()

  for tweet in pos_tweets_list:
    for word in tweet:
      pos_word_list.append(word)
      pos_word_set.add(word)
      total_word_set.add(word)
        

  for tweet in neg_tweets_list:
    for word in tweet:
      neg_word_list.append(word)
      neg_word_set.add(word)
      total_word_set.add(word)
        
    #Total word count
    total_pos = len(pos_word_list)
    total_neg = len(neg_word_list)
    total_count = total_pos + total_neg

    #Total word types
    pos_types = len(pos_word_set)
    neg_types = len(neg_word_set)
    total_types = len(total_word_set) + 1

    #Add to Counter
    pos_counter = Counter(pos_word_list)
    neg_counter = Counter(neg_word_list)

     #Calculate and store the probability that a word is positive or negative
    pos_prob_dict = {}
    neg_prob_dict = {}

    #Store log probabilities to sum later
    pos_sum = []
    neg_sum = []


    #Iterate through unique words, access counts
    #Calculate the likelihood per word
    for word in pos_word_set:

      pos_occurences = pos_counter[word]  #Number of times in positive tweet

      #The probability that a word is positive
      prob_pos = (pos_occurences + laplace) / (total_pos + laplace * (total_types))
      pos_prob_dict.update({word : prob_pos})
      

    for word in neg_word_set:

      neg_occurences = neg_counter[word]

      #The probability that a word is negative
      prob_neg = (neg_occurences + laplace) / (total_neg + laplace * (total_types))
      neg_prob_dict.update({word : prob_neg})
      

    #Calculate probability for unseen positive and negative words
    pos_prob_unseen = (laplace) / (total_pos + laplace * (total_types))
    neg_prob_unseen = (laplace) / (total_neg + laplace * (total_types))
    

    pos_prob_dict.update({"UNK" : pos_prob_unseen})
    neg_prob_dict.update({"UNK" : neg_prob_unseen})

  #Pickle the probabilities to access later
  with open('pos_prob.pkl', 'wb') as fp:
    pickle.dump(pos_prob_dict, fp)

  with open('neg_prob.pkl', 'wb') as fp:
    pickle.dump(neg_prob_dict, fp)
