import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sentiment
from pandas.io.json import json_normalize
import json
from bson import json_util
import pandas as pd
from scipy.special import softmax

sanitized = json.loads(json_util.dumps(sentiment.clean_col.find()))

normalized = pd.json_normalize(sanitized)

df = pd.DataFrame(normalized)

df.sort_values('id', ascending = True, inplace = True)

df=df[["id","text","user.location","Tweet_lemmatized","Tweet_punct"]]

def tweetprocessing(tweet):
       
        tweet = " ".join(tweet)

        # load model and tokenizer
        score = SentimentIntensityAnalyzer().polarity_scores(tweet)
        #print(score)
        #return score
        labels=['Negative', 'Neutral','Positive']
        
        if(score['neg'] > score['pos'] ):
            # text="Negative"  
            # return (text,score['neg'])

            if(score['neg'] >= 0.500 and score['neg'] <= 0.750 ):
                text= "Sorrowful"
                return (text,score['neg'],labels[0])
            elif(score['neg'] >  0.750 ):
                text= "Dead Inside"
                return (text,score['neg'],labels[0])
            else:
                text= "sad"
                return (text,score['neg'],labels[0])              
        elif(score['pos']  > score['neg']):

            #   text="positive"  
            #   return (text,score['pos'])
            if(score['pos'] >= 0.500 and score['pos'] <= 0.750 ):
                text="joyful"
                return (text,score['pos'],labels[2])
            elif(score['pos'] >  0.750 ):
                text= "Ultimate Good"
                return (text,score['pos'],labels[2])
            else:
                text= "Happy" 
                return (text,score['pos'],labels[2]) 

        else:
             text="Neutral"  
             return (text,score['neu'],labels[1])          



        #return (l,s)

df['Tweet_Final'] = df['Tweet_lemmatized'].apply(lambda x: tweetprocessing(x))
#df=df.rename(columns={'Tweet_lemmatized' :'tweet'})

df[['Sentiment', 'Score','Polarity']] = pd.DataFrame(df['Tweet_Final'].tolist(), index=df.index)

print(df['Sentiment'].value_counts())

print(df['Polarity'].value_counts())

df=df.rename(columns={'Tweet_lemmatized' :'tweet'})

df=df[["id","text","tweet","Sentiment","Score","Polarity","Tweet_punct"]]

print(df.head(5))

df.reset_index(inplace=True)

data_dict = df.to_dict("records")

sentiment.temp_col.insert_many(data_dict)