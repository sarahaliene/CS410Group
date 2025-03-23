#CS410 Rough code tweet analysis

import numpy as np
import nltk

#Stopwords
from nltk.corpus import stopwords as stopwords

#Tweet Tokenizer
#from https://www.nltk.org/api/nltk.tokenize.casual.html#module-nltk.tokenize.casual
from nltk.tokenize import TweetTokenizer

#Word stemmer
from nltk.stem.snowball import EnglishStemmer

#Words considered valid words by nltk
from nltk.corpus import words as nltkwords

#NLTK Sentiment Analyzer tool
from nltk.sentiment import sentiment_analyzer

#NLTK basic sentiment analysis utilities
#demo_liu_hu_lexicon(sentence, plot=False) -> uses Liu Hu lexicon for pos/neg/neu word classification
#demo_sent_subjectivity(text) -> classify sentence as subjective or objective
from nltk.sentiment import util as sentiment_util

#For counting occurences of things in general
from collections import Counter

#These may need to be downloaded?
nltk.download('punkt_tab')
nltk.download('stopwords')

#Use stemmer.stem(word) to stem word
stemmer = EnglishStemmer()