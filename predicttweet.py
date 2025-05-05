import pickle
import math


def predict_tweet(tweet):
  #Unpickle? the files needed for computations

  pos_prob_file = open('pos_prob.pkl', 'rb')    
  pos_prob = pickle.load(pos_prob_file)

  neg_prob_file = open('neg_prob.pkl', 'rb')    
  neg_prob = pickle.load(neg_prob_file)

  pos_prior = 0.5
  neg_prior = 1.0 - pos_prior

  pos_words = []
  neg_words = []

  for word in tweet:

    if word in pos_prob:
      prob_pos = pos_prob.get(word)
    else:
      prob_pos = pos_prob.get("UNK")

    if word in neg_prob:
      prob_neg = neg_prob.get(word)
    else:
      prob_neg = neg_prob.get("UNK")

    pos_words.append(prob_pos)
    neg_words.append(prob_neg)

  #Calculate posterior probability
    pos_post = 0
    for value in pos_words:
      pos_post += math.log10(value)
    
    pos_post += math.log10(pos_prior)

    neg_post = 0
    for value in neg_words:
      neg_post += math.log10(value)

    neg_post += math.log10(neg_prior)

    if pos_post >= neg_post:
      return("positive")
    else:
      return("negative")
