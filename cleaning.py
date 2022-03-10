from matplotlib.pyplot import text
import pandas as pd

import sentiment1

import pandas as pd
from bson import json_util, ObjectId
from pandas.io.json import json_normalize
import json

import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
set(stopwords.words('english'))
from wordcloud import WordCloud
nltk.download('punkt')
import string
nltk.download('wordnet')
nltk.download('omw-1.4')




sanitized = json.loads(json_util.dumps(sentiment1.raw_col.find()))
normalized = pd.json_normalize(sanitized)
df = pd.DataFrame(normalized)
print(df.head(2))
print(df.columns)
df.sort_values('id', ascending = True, inplace = True)
print(df.head(3))
df=df[["id","text","user.location"]]
df=df[df['user.location'].str.contains('USA',na=False)]
print(df.head(5))

print(df.shape)



#df['clean_tweets'] = ", ".join(df['text'].astype(str))
#df['clean_tweets']=df.loc[df['text'].notna(), "other_studio_emp"] = ",".join(df['text'])

#df.clean_tweets = df.clean_tweets.str.lower()
print(df.head(2))


# df.clean_tweets = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', df.clean_tweets)

# clean_tweets = re.sub(r'#([^\s]+)', r'\1', clean_tweets) 

# # remove usernames
# df['clean_tweets'] = re.sub('@[^\s]+', 'AT_USER', clean_tweets)
# clean_tweets = re.sub('@ [^\s]+', 'AT_USER', clean_tweets)
# #replace consecutive non-ASCII characters with a space
# clean_tweets = re.sub(r'[^\x00-\x7F]+',' ', clean_tweets)
# # remove punctuation
# clean_tweets = re.sub(r'[^\w\s]', '', clean_tweets)
# # remove emojis
# clean_tweets = clean_tweets.encode('ascii', 'ignore').decode('ascii')
# # remove trailing spaces
# clean_tweets = clean_tweets.strip()
# #remove numbers
# clean_tweets = re.sub('[0-9]+', '', clean_tweets)
# # tokenize 
# clean_tweets = word_tokenize(clean_tweets)
# #remove stop words
# stop = stopwords.words('english')
# clean_tweets = [w for w in clean_tweets if not w in stop]
# print(clean_tweets)

# print(df.head(3))

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

df['Tweet_punct'] = df['text'].apply(lambda x: remove_punct(x.lower()))
print(df.head(5))

def tokenization(text):
    text = re.split('\W+', text)
    return text

df['Tweet_tokenized'] = df['Tweet_punct'].apply(lambda x: tokenization(x.lower()))
print(df.head())


stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
     text = [word for word in text if word not in stopword]
     return text
    
df['Tweet_nonstop'] = df['Tweet_tokenized'].apply(lambda x: remove_stopwords(x))
print(df['Tweet_nonstop'].head(10))

ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

df['Tweet_stemmed'] = df['Tweet_nonstop'].apply(lambda x: stemming(x))
print(df.head(3))

wn = nltk.WordNetLemmatizer()

def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text

df['Tweet_lemmatized'] = df['Tweet_nonstop'].apply(lambda x: lemmatizer(x))
print(df.head())


df.reset_index(inplace=True)
# maxVal = 5
# df.where(df >= maxVal, maxVal)
data_dict = df.to_dict("records")
sentiment1.clean_col.insert_many(data_dict) 



# ##Applying model


# from socket import timeout

# import tweepy
# from tweepy import OAuthHandler 
# from tweepy import Stream
# import pandas as pd
# import json
# import pymongo
# from pymongo import MongoClient
# import time
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from scipy.special import softmax

# def tweetprocessing(tweet):

#     tweet_proc = " ".join(tweet)

#     # load model and tokenizer
#     roberta = "cardiffnlp/twitter-roberta-base-sentiment"

#     model = AutoModelForSequenceClassification.from_pretrained(roberta)
#     tokenizer = AutoTokenizer.from_pretrained(roberta)

#     labels = ['Negative', 'Neutral', 'Positive']

#     # sentiment analysis
#     encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')

#     output = model(**encoded_tweet)

#     scores = output[0][0].detach().numpy()
#     scores = softmax(scores)

#     print(type(scores))
#     if(scores[0] > scores[1] and scores[0] > scores[2]):

#         if(scores[0] >= 0.50 and scores[0] <= 0.75 ):
#             return "Negative"
#         elif(scores[0] >  0.75 ):
#             return "very Negative"       
#     elif (scores[1] > scores[0] and scores[1] > scores[2]):
#         if(scores[1] >= 0.50 and scores[1] <= 0.75):
#             return "Neutral"
#         elif(scores[1] >  0.75 ):
#             return "very Neutral"
#         print("neutral")
    
#     elif (scores[2] > scores[0] and scores[2] > scores[1]):
#         if(scores[2] >= 0.50 and scores[2] <= 0.75 ):
#             return "Positive"
#         elif(scores[2] >  0.75 ):
#             return "very Positive"
 
# df['Tweet_final'] = df['Tweet_lemmatized'].head(4).apply(lambda x: tweetprocessing(x))
# print(df.head(4))







#     # def clean_tweets(tweets):
#     #     #try:
#     #         # Preprocessing
#     #         clean_tweets = ' '.join(tweets)
#     #         # convert to lower case
#     #         clean_tweets =  clean_tweets.lower()
#     # remove URLs
#     # clean_tweets = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', clean_tweets) 
#     # # remove the # in #hashtag
#     # clean_tweets = re.sub(r'#([^\s]+)', r'\1', clean_tweets) 
#     # # remove usernames
#     # clean_tweets = re.sub('@[^\s]+', 'AT_USER', clean_tweets)
#     # #replace consecutive non-ASCII characters with a space
#     # clean_tweets = re.sub(r'[^\x00-\x7F]+',' ', clean_tweets)
#     # # remove punctuation
#     # clean_tweets = re.sub(r'[^\w\s]', '', clean_tweets)
#     # # remove emojis
#     # clean_tweets = clean_tweets.encode('ascii', 'ignore').decode('ascii')
#     # # remove trailing spaces
#     # clean_tweets = clean_tweets.strip()
#     # remove numbers
#     #clean_tweets = re.sub('[0-9]+', '', clean_tweets)
#     # tokenize 
#     #clean_tweets = word_tokenize(clean_tweets)
#     # # remove stop words
#     # stop = stopwords.words('english')

#     # clean_tweets = [w for w in clean_tweets if not w in stop] 

#     #         return clean_tweets
#     # df['Clean_text'] = df['text'].apply(lambda x: clean_tweets(x.lower()))



#     # print(df['Clean_text'].head(5))       

