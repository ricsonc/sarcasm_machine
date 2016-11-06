import numpy as np
import gensim,logging
import sys
import re
import os
from os.path import join

#import sarc as parser


tweet_model = gensim.models.Word2Vec.load_word2vec_format('glove.converted.txt', binary=False)
#general_model = Word2Vec.load('glove.twitter.27B.25d.txt')



def get_dim(word,model=tweet_model):
	if word in model:
		dim = np.append(model[word],0)
	else:
		dim = np.append(np.zeros(25),1)
	return dim

def get_matrix(sentence,model=tweet_model):
	sentence = parse(sentence)
	length = len(sentence)
	matrix = np.zeros(shape=(length,26)) #26 is the word dim + dummy dim for if the word in dict
	for i in xrange(length):
		dim = get_dim(sentence[i],model = model)
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

'''
testing_file = 'corpus/Ironic/1_1_R280644F3NWFFN.txt'
f = open(testing_file,'r')
print get_matrix(f,1)
'''
def main():
	ironic = os.listdir('corpus/Ironic')
	regular = os.listdir('corpus/Regular')
	corpus_list = []
	label_list = []
	for ironicFile in ironic:
		if ironicFile.endswith('.txt'): 
			f = open('corpus/Ironic/' + ironicFile)
			matrix = get_matrix(f)
			f.close()
			corpus_list.append(matrix)
			label_list.append(1)
	for regularFile in regular:	
		if regularFile.endswith('.txt'):
			f = open('corpus/Regular/' + regularFile)
			matrix = get_matrix(f)
			f.close()
			corpus_list.append(matrix)
			label_list.append(0)
	return corpus_list,label_list

corpus_list,label_list = main()
np.save('corpus_list',corpus_list)
np.save('label_list',label_list)
