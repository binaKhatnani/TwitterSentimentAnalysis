from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

#from matplotlib.pyplot import text
import pandas as pd

import sentiment1

import pandas as pd
from bson import json_util, ObjectId
from pandas.io.json import json_normalize
import json

import pandas as pd
import re




sanitized = json.loads(json_util.dumps(sentiment1.clean_col.find()))
normalized = pd.json_normalize(sanitized)
df = pd.DataFrame(normalized)
df.sort_values('id', ascending = True, inplace = True)
df=df[["id","text","user.location","Tweet_lemmatized"]]

def tweetprocessing(tweet):
       

        tweet_proc = " ".join(tweet)

        # load model and tokenizer
        roberta = "cardiffnlp/twitter-roberta-base-sentiment"

        model = AutoModelForSequenceClassification.from_pretrained(roberta)
        tokenizer = AutoTokenizer.from_pretrained(roberta)

        labels = ['Negative', 'Neutral', 'Positive']

        # sentiment analysis
        encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
        # output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
        output = model(**encoded_tweet)

        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        
        if(scores[0] > scores[1] and scores[0] > scores[2]):
            # return scores[2] 
            # print('Negative')
            if(scores[0] >= 0.50 and scores[0] <= 0.75 ):
                text= "Sorrowful"
                return (text,scores[0])
            elif(scores[0] >  0.75 ):
                text= "Dead Inside"
                return (text,scores[0])
            elif(scores[0] <= 0.50 ):
                text= "sad"
                return (text,scores[0])          
        elif (scores[1] > scores[0] and scores[1] > scores[2]):
             text="Sab Changa Si"  
             return (text,scores[1])    
        elif (scores[2] > scores[0] and scores[2] > scores[1]):
            if(scores[2] >= 0.50 and scores[2] <= 0.75 ):
                text="joyful"
                return (text,scores[2])
            elif(scores[2] >  0.75 ):
                text= "Ultimate Good"
                return (text,scores[2])
            elif(scores[2] <= 0.50 ):
                text= "Happy" 
                return (text,scores[2])   



        #return (l,s)
df['Tweet_Final'] = df['Tweet_lemmatized'].head(4).apply(lambda x: tweetprocessing(x))
df=df.rename(columns={'Tweet_lemmatized' :'tweet'})

df[['Sentiment', 'Score']] = pd.DataFrame(df['Tweet_Final'].tolist(), index=df.index)
df=df[["id","text","tweet","Sentiment","Score"]]
print(df.columns)

