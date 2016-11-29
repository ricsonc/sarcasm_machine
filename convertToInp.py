import numpy as np
import gensim,logging
import sys
import re
import os
from os.path import join
from textblob import TextBlob

#import sarc as parser


tweet_model = gensim.models.Word2Vec.load_word2vec_format('glove.converted.txt', binary=False)
#general_model = Word2Vec.load('glove.twitter.27B.25d.txt')



def get_dim(word,model=tweet_model):
	if word in model:
		dim = np.append(model[word],0)
	else:
		dim = np.append(np.zeros(25),1)
	return dim

def get_polarity_word(word):
	blob = TextBlob(word)
	polarity = blob.sentiment.polarity
	return polarity

def get_polarity_vector(sentence):
	sentence = parse(sentence)
	length = len(sentence)
	vector = np.zeros(shape=(length,1))
	for i in xrange(length):
		pol = get_polarity_word(sentence[i])
		vector[i] = pol
	return vector

def get_polarity_vector_twitter(sentence):
	sentence = parse_twitter(sentence)
	length = len(sentence)
	vector = np.zeros(shape=(length,1))
	for i in xrange(length):
		pol = get_polarity_word(sentence[i])
		vector[i] = pol
	return vector

def get_matrix(sentence,model=tweet_model):
	sentence = parse(sentence)
	length = len(sentence)
	matrix = np.zeros(shape=(length,26)) #26 is the word dim + dummy dim for if the word in dict
	for i in xrange(length):
		dim = get_dim(sentence[i],model = model)
		matrix[i] = dim
	return matrix

def get_n_gram_matrix(sentence_input,n,model=tweet_model):
	sentence = parse(sentence_input)
	length = len(sentence)-n+1
	if length<=0:
		return get_matrix(sentence_input)
	matrix = np.zeros(shape=(length,26))
	for i in xrange(length):
		dim = get_dim(sentence[i],model =model)
		for ngram in xrange(n-1):
			dim+=get_dim(sentence[i+ngram],model=model)
		matrix[i] = dim
	return matrix

def parse(inp):
	#Does not cosinder punctuation
	reviewFlag = 0
	parsed_inp = []
	for row in inp:
		# print row
		# look for the start of the actual review...
		if row == '<REVIEW>\n':
			reviewFlag = 1
			continue
			# print 'hi'
		# then start parsing
		if reviewFlag == 1:
			if row != '</REVIEW>':
				line = re.split(' |\.|\,|\!|\?|\/|\(|\)|\-|\_|\\\\|\@|\#|\$|\%|\^|\&|\*|\=|\+|\[|\]|\{|\}|\;|\:|\>|\<|\~', row)
				# print line
				for word in line:
					# if word.find('emoticon') != -1:
						# print word
					word = word.lower().strip('`~!@#$%^&*()-_=+[]}{\\|\'\";:/?.>,<\n')
					if word != '':
						parsed_inp.append(word)
	return parsed_inp

def parse_twitter(inp):
	parsed_inp = []
	line = re.split(' |\.|\,|\!|\?|\/|\(|\)|\-|\_|\\\\|\$|\%|\^|\&|\*|\=|\+|\[|\]|\{|\}|\;|\:|\>|\<|\~', inp)
	print line
	for word in line:
		# if word.find('emoticon') != -1:
			# print word
		word = word.lower().strip('`~!$%^&*()-_=+[]}{\\|\'\";:/?.>,<\n\r')
		# we changed @ behavior. also changed # behavior, but
		# that might be okay since some # are presumably
		# worth treating as real words. but strip out #sarcasm since it was
		# used to generate the corpus and thus the trivial classfier weights
		# #sarcasm as 1 and everything else as 0
		if word != '' and word != '#sarcasm':
			if word[0] == '@':
				parsed_inp.append('@')
			elif word[0] == '#':
				parsed_inp.append('#')
				if len(word) > 1:
					parsed_inp.append(word[1:])
			else:
				parsed_inp.append(word)
	return parsed_inp

'''
testing_file = 'corpus/Ironic/1_1_R280644F3NWFFN.txt'
f = open(testing_file,'r')
print get_matrix(f,1)
'''
def main(ngram=0):
	ironic = os.listdir('corpus/Ironic')
	regular = os.listdir('corpus/Regular')
	corpus_list = []
	label_list = []
	for ironicFile in ironic:
		if ironicFile.endswith('.txt'): 
			f = open('corpus/Ironic/' + ironicFile)
			if ngram!=0:
				matrix = get_n_gram_matrix(f,ngram)
			else:
				matrix = get_matrix(f)
			f.close()
			corpus_list.append(matrix)
			label_list.append(1)
	for regularFile in regular:	
		if regularFile.endswith('.txt'):
			f = open('corpus/Regular/' + regularFile)
			if ngram!=0:
				matrix = get_n_gram_matrix(f,ngram)
			else:
				matrix = get_matrix(f)
			f.close()
			corpus_list.append(matrix)
			label_list.append(0)
	return corpus_list,label_list

def main_polarity():
	ironic = os.listdir('corpus/Ironic')
	regular = os.listdir('corpus/Regular')
	corpus_list = []
	for ironicFile in ironic:
		if ironicFile.endswith('.txt'): 
			f = open('corpus/Ironic/' + ironicFile)
			vector = get_polarity_vector(f)
			f.close()
			corpus_list.append(vector)
	for regularFile in regular:	
		if regularFile.endswith('.txt'):
			f = open('corpus/Regular/' + regularFile)
			vector = get_polarity_vector(f)
			f.close()
			corpus_list.append(vector)
	return corpus_list

def main_polarity_twitter():
	ironicFile = open('twitDB_sarcasm.csv')
	regularFile = open('twitDB_ragular.csv')
	corpus_list = []
	for line in ironicFile:
		vector = get_polarity_vector_twitter(line)
		corpus_list.append(vector)
	for line in regularFile:
		vector = get_polarity_vector_twitter(line)
		corpus_list.append(vector)
	ironicFile.close()
	regularFile.close()

	# for ironicFile in ironic:
	# 	if ironicFile.endswith('.txt'): 
	# 		f = open('corpus/Ironic/' + ironicFile)
	# 		vector = get_polarity_vector(f)
	# 		f.close()
	# 		corpus_list.append(vector)
	# for regularFile in regular:	
	# 	if regularFile.endswith('.txt'):
	# 		f = open('corpus/Regular/' + regularFile)
	# 		vector = get_polarity_vector(f)
	# 		f.close()
	# 		corpus_list.append(vector)
	return corpus_list


#corpus_list_polarity_word = main_polarity()
#np.save('corpus_list_polarity_word',corpus_list_polarity_word)

twitter_corpus_list_polarity_word = main_polarity_twitter()
np.save('twitter_corpus_list_polarity_word',twitter_corpus_list_polarity_word)


corpus_list,label_list = main()
#np.save('corpus_list',corpus_list)
np.save('label_list',label_list)


#corpus_list_bigram,label_list = main(ngram=2)
#np.save('corpus_list_bigram',corpus_list_bigram)
#corpus_list_trigram,label_list = main(ngram=3)
#np.save('corpus_list_trigram',corpus_list_trigram)