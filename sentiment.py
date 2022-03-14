from socket import timeout

import tweepy
from tweepy import OAuthHandler 
from tweepy import Stream
import pandas as pd
import json
import pymongo
from pymongo import MongoClient
import time

#Get the mongodb uri details

myclient = pymongo.MongoClient("mongodb+srv://twittersentimentanalysis:Twitter01@cluster0.n2bao.mongodb.net/twitterSentiment?retryWrites=true&w=majority")

mydb = myclient["twitterSentiment"]

raw_col = mydb["raw"]

clean_col=mydb["clean"] 

ML_col=mydb["ML"]

temp_col=mydb["temp"]

class StdOutListener(Stream):
    
    #set the limit on number of tweets to consumed 
    limit=50000

    tweets=[]
        
    def on_data(self, data):
         
        self.tweets.append(json.loads(data))
        
        if len(self.tweets) == self.limit:
            print(len(self.tweets),"\n")
            print("\n   tweet completed   ")
            self.disconnect()
                
                      

    def on_error(self, status_code):
    
        if status_code == 420:
            #returning False in on_error disconnects the stream
            return False

if __name__ == "__main__": 

    #Creates the table for storing the tweets
    term_to_search = ["#bitcoin","#Russia","#covid19"]

    twitter_stream = StdOutListener(
    "G7rqMw6UMwSUjN5jPiRKG2FM9", "LDiDUphfFz5HFv02JtLX5cbr28bajggT8lU3fiQjW8EhvGb6Hf",
    "86975362-g7oY9QYJv7dee3ce7VdZ6mwzYSqq2vGwqtvp3nkdl", "EVO09AaQLldW8FhIysO7bgZMKOjqecbtVHCJbLFh2EoSv")

    twitter_stream.filter(languages=["en"],track=term_to_search)

    raw_col.insert_many(twitter_stream.tweets)