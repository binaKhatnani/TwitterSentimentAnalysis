from matplotlib.pyplot import text
import pandas as pd

import sentiment
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

sanitized = json.loads(json_util.dumps(sentiment.raw_col.find()))

normalized = pd.json_normalize(sanitized)


df = pd.DataFrame(normalized)

print(df.head(2))

print(df.columns)

df.sort_values('id', ascending = True, inplace = True)

print(df.head(3))

df=df[["id","text","user.location"]]

df=df[df['user.location'].str.contains('null',na=False)]

print(df.head(20))

print(df.shape)

print(df.head(2))

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

data_dict = df.to_dict("records")

sentiment.clean_col.insert_many(data_dict)