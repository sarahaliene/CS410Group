import pickle
import math
#is functioning properly as of 5-5-25

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

    prob_pos = pos_prob.get(word, 0)
    prob_neg = neg_prob.get(word, 0)

    if prob_pos == 0:
      prob_pos = pos_prob.get("UNK")
      pos_words.append(prob_pos)

    else:
      pos_words.append(prob_pos)

    if prob_neg == 0:
      prob_neg = neg_prob.get("UNK")
      neg_words.append(prob_neg)

    else:
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


#pred = predict_tweet("weird prediction here :)")
#print(pred)
