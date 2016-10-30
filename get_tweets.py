from twython import Twython, TwythonError
import numpy as np
import gensim,logging
from gensim.models import Word2Vec

'''
MAX_ATTEMPTS = 400
tweets = []

APP_KEY = 't8UAnOlePi3XIiB8RBFF4jQ8a'
APP_SECRET = 'y8BPGdcHt0WSk2R5C3LoJdQT5a7Glq2G8As06rVfSaNyO8PeXv'
twitter = Twython(APP_KEY, APP_SECRET, oauth_version=2)
ACCESS_TOKEN = twitter.obtain_access_token()
print ACCESS_TOKEN


twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)



for i in xrange(MAX_ATTEMPTS):
	if (i == 0):
		result = twitter.search(q='#sarcasm AND -filter:retweets,', lang='en',count = 100)
	else:
		print i,next_max_id
		result = twitter.search(q='#sarcasm AND -filter:retweets', lang='en',count = 100, include_entities = 'true',max_id = next_max_id)
	for tweet in result['statuses']:
		tweet_text = tweet['text'].encode('utf-8')
		tweets.append(tweet_text)
	print result['search_metadata']
	try:
		next_result_url_params = result['search_metadata']['next_results']
		next_max_id = next_result_url_params.split('max_id=')[1].split('&')[0]
	except:
		break
tempDict = dict()
tempDict['tweets'] = tweets
#result = twitter.search(q='#sarcastic AND -filter:retweets AND -filter:replies', lang='en',count = 100)
np.save('temp.npy',tempDict)
'''

result = np.load('temp.npy')


count = 0
tweets = result
tweets_no_pic = []
tweets_no_pic_parsed = []
for i in tweets:
	if 'http' not in i:
		count+=1
		tweets_no_pic.append(i)
		tweets_no_pic_parsed.append(i.split())


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = Word2Vec(tweets_no_pic_parsed)
print model['fantasy']
