from collections import Counter

def create_vocab(tweet_list, n):

	tweet_counter = Counter()

	#Add each line (tweet) to the counter
	for tweet in tweet_list:
		tweet_counter.update(tweet)


	top_200 = tweet_counter.most_common(n)


	return top_200
