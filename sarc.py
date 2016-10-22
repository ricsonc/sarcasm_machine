import csv
import sys

# sketchy temp fix
csv.field_size_limit(sys.maxsize)

sarcDict = {}
notSarcDict = {}

sw = open('ml_hw1/hw1_dataset_nb/sw.txt', 'r')
stopWords = [x.strip('\n').lower() for x in sw.readlines()]
sw.close()

rowNum = 0
with open('sarcasm_v2.csv', 'rb') as csvfile:
	sarcReader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in sarcReader:
		if rowNum == 0:
			rowNum += 1
			continue
		if rowNum == 1:
			print row
			print row[4]
			print row[5].split(' ')
		# # print rowNum
		# rowNum += 1

	# Naive one-word sarcasm dictionary
		comment = row[5].split(' ')

		# Search sarcastic reviews
		if row[1] == 'sarc':
			# THIS MAY BE WRONG -- POSSIBLY we only want the response text
			for word in comment:
				word = word.lower().strip('`~!@#$%^&*()-_=+[]}{\\|\'\";:/?.>,<\n')
				if word not in stopWords:
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
				if word not in stopWords:
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

#	print 'SARCDICT'
#	print sarcDict
#	print 'NOTSARCDICT'
#	print notSarcDict