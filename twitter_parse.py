import os
import re

def parse_amazon(inp):
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

# ironic = os.listdir('corpus/Ironic')
# regular = os.listdir('corpus/Regular')
# corpus_list = []
# for ironicFile in ironic:
# 	if ironicFile.endswith('.txt'): 
# 		f = open('corpus/Ironic/' + ironicFile)
# 		parsed = parse_amazon(f)

# print 'A parsed Amazon review:'
# print parsed

# we want this to be given a line, which will be a tweet
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

ironicFile = open('twitDB_sarcasm.csv')
# for line in ironicFile:
# 	parsed = parse_twitter(line)
# 	print parsed
line = ironicFile.readline()
print parse_twitter(line)
line = ironicFile.readline()
print parse_twitter(line)
ironicFile.close()

# lab_user@CSB-CVC-00:~/Downloads/sarcasm_machine$ python twitter_parse.py 
# ['while', 'from', 'a', 'purely', 'academic', 'standpoint', 'this', 'book', 'is', 'informative', 'interesting', 'and', 'overall', 'excellently', 'written', 'it', 'has', 'a', 'far', 'greater', 'purpose', 'than', 'simply', 'to', 'satisfy', 'idle', 'curiosity', 'monkeys', 'and', 'apes', 'exist', 'as', 'one', 'of', 'the', 'greatest', 'threats', 'to', 'mankind', 'i', 'submit', 'as', 'evidence', 'the', 'multiple', 'cases', 'of', 'chimps', 'just', 'flipping', 'out', 'and', 'going', 'bananas', 'on', 'their', 'owners', 'often', 'biting', 'off', 'noses', 'fingers', 'and', 'testes', 'or', 'even', 'killing', 'their', 'comparatively', 'helpless', 'victims', 'a', '90', 'pound', 'chimp', 'is', 'more', 'than', 'a', 'match', 'for', 'most', 'fully', 'grown', 'men', 'capuchins', 'while', 'seemingly', 'innocuous', 'due', 'to', 'their', 'diminutive', 'stature', 'and', 'cute', 'appearance', 'are', 'possibly', 'the', 'worst', 'of', 'the', 'bunch', 'disturbingly', 'self', 'aware', 'capuchins', 'as', 'described', 'in', 'this', 'book', 'are', 'probably', 'the', 'most', 'intelligent', 'of', 'the', 'new', 'world', 'monkeys', 'possessing', 'exceptionally', 'large', 'brains', 'for', 'their', 'body', 'size', 'second', 'only', 'to', 'humans', 'in', 'addition', 'michelle', 'press', 'does', 'an', 'excellent', 'job', 'of', 'describing', 'some', 'of', 'the', 'more', 'frightening', 'actions', 'of', 'these', 'simian', 'killing', 'machines', 'including', 'forming', 'totem', 'poles', 'of', 'up', 'to', 'four', 'monkeys', 'piled', 'on', 'top', 'of', 'each', 'other', 'as', 'they', 'converge', 'on', 'their', 'doomed', 'prey', 'i', 'encourage', 'anyone', 'at', 'all', 'concerned', 'about', 'the', 'growing', 'threat', 'of', 'capuchin', 'dominance', 'to', 'read', 'this', 'book', 'in', 'order', 'to', 'stem', 'the', 'tide', 'of', 'the', 'marmoset', 'menace', 'the', 'next', 'person', 'they', 'go', 'ape', 'on', 'could', 'be', 'you']
