from socket import timeout

import tweepy
from tweepy import OAuthHandler 
from tweepy import Stream
import pandas as pd
import json
import pymongo
from pymongo import MongoClient
import time



myclient = pymongo.MongoClient("mongodb+srv://twittersentimentanalysis:Twitter01@cluster0.n2bao.mongodb.net/twitterSentiment?retryWrites=true&w=majority")
mydb = myclient["twitterSentiment"]
raw_col = mydb["raw"]
clean_col=mydb["clean"] 

class StdOutListener(Stream):

    # def __init__(self):
    #     super(StdOutListener, self).__init__()
    
    
    limit=10
    tweets=[]
        

    def on_data(self, data):
         
        self.tweets.append(json.loads(data))
        if len(self.tweets) == self.limit:
            print(len(self.tweets),"\n")
            print("\n   tweet completed   ")
            #self.disconnect()
                
              
           
                        #self.disconnect()

                #df=pd.json_normalize([self.receive_tweets])       
                #df.reset_index(inplace=True)
                #print(df)
                #data_dict = df.to_dict("records")
                            # Insert collection
                #mycol.insert_many(data_dict)
                
                
                #     self.pbar.close() # Closes the instance of the progress bar.
                
                     #return True # Closes the stream.
                            
        # except Exception as e:
        #         print(e)

    
    
    
# # Each Tweet object has default id and text fields
        # for tweet in tweets:
        #       a = tweet
        #       tweet_list.append(a)
        #       tweet_df = pd.DataFrame(tweet_list)
        #       print(tweet_df.head(1))
        # return True
# tweet_df = pd.DataFrame(tweet_list)
# print(tweet_df)
        #df=pd.DataFrame([data])
        #df.head(1)
        

    def on_error(self, status_code):
        if status_code == 420:
            #returning False in on_error disconnects the stream
            return False




if __name__ == "__main__": 

    #Creates the table for storing the tweets
    term_to_search = "#bitcoin"
    twitter_stream = StdOutListener(
    "G7rqMw6UMwSUjN5jPiRKG2FM9", "LDiDUphfFz5HFv02JtLX5cbr28bajggT8lU3fiQjW8EhvGb6Hf",
    "86975362-g7oY9QYJv7dee3ce7VdZ6mwzYSqq2vGwqtvp3nkdl", "EVO09AaQLldW8FhIysO7bgZMKOjqecbtVHCJbLFh2EoSv")
    twitter_stream.filter(languages=["en"],track=term_to_search)
    
    # data=[]
    # for tweet in twitter_stream.tweets:
    #     data.append([tweet])
    raw_col.insert_many(twitter_stream.tweets)
    
    #df=pd.json_normalize([twitter_stream.tweets])       
    # df.reset_index(inplace=True)
                
    # data_dict = df.to_dict("records")
    #                         # Insert collection
    # mycol.insert_many(data_dict)   
    
    
    






