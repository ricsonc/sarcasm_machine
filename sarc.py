#!/usr/bin/env python2

#parsing script

import csv
import sys
import re
import os
# import enchant

# FUTURE THINGS: spell checker? splitting words that contain punctuation?

# stop words from hw1, recommended for all parsing
sw = open('sw.txt', 'r')
stopWords = [x.strip('\n').lower() for x in sw.readlines()]
sw.close()

# parsing for newer corpus (corpus.tar, which as of now needs to be
# pre-extracted for this to work)

sarcDict2 = {}
notSarcDict2 = {}

ironic = os.listdir('corpus/Ironic')
regular = os.listdir('corpus/Regular')
# txtCheck = re.compile('*.txt');

for ironicFile in ironic:
	if ironicFile.endswith('.txt'):
		# print ironicFile
		f = open('corpus/Ironic/' + ironicFile)
		reviewFlag = 0
		for row in f:
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
						if word not in stopWords and word != '':
							if word in sarcDict2:
								sarcDict2[word] += 1
							else:
								sarcDict2[word] = 1
		f.close

for regularFile in regular:
	if regularFile.endswith('.txt'):
		# print ironicFile
		f = open('corpus/Regular/' + regularFile)
		reviewFlag = 0
		for row in f:
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
						if word not in stopWords and word != '':
							if word in notSarcDict2:
								notSarcDict2[word] += 1
							else:
								notSarcDict2[word] = 1
		f.close

# print 'SARCDICT_corpus'
# print sarcDict2
# print 'NOTSARCDICT_corpus'
# print notSarcDict2

# parsing for older corpus (sarcasm_v2.txt)

# sketchy temp fix
csv.field_size_limit(sys.maxsize)

sarcDict = {}
notSarcDict = {}

rowNum = 0
with open('sarcasm_v2.csv', 'rb') as csvfile:
	sarcReader = csv.reader(csvfile, delimiter=',', quotechar='\"')
	for row in sarcReader:
		if rowNum == 0:
			rowNum += 1
			continue
		# if rowNum == 70:
		#  	print row
		#  	print row[4]
		#  	print row[4].split(' ')
		#  	print re.split(' |\.|\,|\!|\?', row[4])
		# # print rowNum
		rowNum += 1

	# Naive one-word sarcasm dictionary
		# comment = re.split(' ', row[4])
		comment = re.split(' |\.|\,|\!|\?|\/|\(|\)|\-|\_|\\\\|\@|\#|\$|\%|\^|\&|\*|\=|\+|\[|\]|\{|\}|\;|\:|\>|\<|\~', row[4])

		# Search sarcastic reviews
		if row[1] == 'sarc':
			# THIS MAY BE WRONG -- POSSIBLY we only want the response text
			for word in comment:
				# if word.find('emoticon') != -1:
					# print word
				word = word.lower().strip('`~!@#$%^&*()-_=+[]}{\\|\'\";:/?.>,<\n')
				if word not in stopWords and word != '':
					if word in sarcDict:
						sarcDict[word] += 1
					else:
						sarcDict[word] = 1
				# for word in row[1:]:
				# 	word = word.lower().strip('`~!@#$%^&*()-_=+[]}{\\|\'\";:/?.>,<\n')
				# 	if word not in stopWords:
				# 		if word in sarcDict:
				# 			sarcDict[word] += 1
				# 		else:
				# 			sarcDict[word] = 1

		# Search non-sarcastic reviews
		if row[1] == 'notsarc':
			for word in comment:
				word = word.lower().strip('`~!@#$%^&*()-_=+[]}{\\|\'\";:/?.>,<\n')
				if word not in stopWords and word != '':
					if word in notSarcDict:
						notSarcDict[word] += 1
					else:
						notSarcDict[word] = 1
			# for word in row[1:]:
			# 	word = word.lower().strip('`~!@#$%^&*()-_=+[]}{\\|\'\";:/?.>,<\n')
			# 	if word not in stopWords:
			# 		if word in notSarcDict:
			# 			notSarcDict[word] += 1
			# 		else:
			# 			notSarcDict[word] = 1

# print 'SARCDICT_sarcasm_v2.csv'
# print sarcDict
# print 'NOTSARCDICT_sarcasm_v2.csv'
# print notSarcDict
