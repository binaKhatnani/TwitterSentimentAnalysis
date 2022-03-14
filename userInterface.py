import streamlit as st
from PIL import Image

image = Image.open('/Users/binakhatnani/Desktop/mlflow.png')

#image_mlflow = Image.open('/Users/binakhatnani/Desktop/mlflow.png')

header=st.container()
dataset=st.container()
model_build=st.container()
ml_flow=st.container()
model_testing=st.container()
modeltraining=st.container()


with header:
    st.title('Twitter Sentiment Analysis')
    st.text('In this project we have collected live tweets using twitter api and analyzed\nto get the sentiments of the user tweets.')

with dataset:
    import json
    from bson import json_util
    import sentiment
    import pandas as pd
    st.header('Data collection and Data Preprocessing')
    st.text('Raw data obtained from twitter was preprocessed using nltk library\nand then stored in MongoDB Atlas.')
    st.image(image, caption='MongoDB collections')
    sanitized = json.loads(json_util.dumps(sentiment.ML_col.find()))
    normalized = pd.json_normalize(sanitized)
    df = pd.DataFrame(normalized)
    st.write(df.head(5))

with model_build:
    st.header('Machine learning Model')
    st.text('SentimentIntensityAnalyzer from nltk library, model was used to extract the\npolarity of tweets and the the polarity was further classified to get the sentiments.\nThe output was stored in MongoDB Atlas for further training the model.Logistic\nregression model was created to train model with the extracted data and polarity.\nThe model can be used to manually enter a text and predict the sentiment.')
   
with ml_flow:
    st.header('MLFLOW Containerization')
    st.text('The trained Logistic Model is then containerized in mlflow.')
    st.image(image, caption='Mlflow regsitered model versions')

with model_testing:
    st.header('Testing the model by end user.')
    st.text('A user interface is created using streamlit.\nThe end users can manually enter a text to get its sentiment.')

with modeltraining:

    st.header('Test the model')
    with st.form(key='Enter text'):
      
        input_text = st.text_input('Enter your tweet')
        submit_button = st.form_submit_button(label='Submit')
        
        
        if submit_button:

            import sentiment
            
            import pandas as pd
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.ensemble import RandomForestClassifier
            import logging
            import os
            import warnings
            import sys

            logging.basicConfig(level=logging.WARN)
            logger = logging.getLogger(__name__)

            sanitized = json.loads(json_util.dumps(sentiment.temp_col.find()))
            normalized = pd.json_normalize(sanitized)
            df = pd.DataFrame(normalized)
            df.sort_values('id', ascending = True, inplace = True)
            df = df[pd.notnull(df['text'])]

            df['Polarity_int'] = df['Polarity'].factorize()[0]

            polarity_id_df = df[['Polarity', 'Polarity_int']].drop_duplicates().sort_values('Polarity_int')
            polarity_to_id = dict(polarity_id_df.values)
            id_to_polarity = dict(polarity_id_df[['Polarity_int', 'Polarity']].values)
            print(df["Polarity_int"])
            
            from sklearn.feature_extraction.text import TfidfVectorizer

            tfidf = TfidfVectorizer(sublinear_tf=True, min_df=10, norm='l2', encoding='latin-1', ngram_range=(1,10), stop_words='english')

            features = tfidf.fit_transform(df['text']).toarray()
            labels = df['Polarity_int']
            features.shape

            from sklearn.feature_selection import chi2
            import numpy as np

            N = 100000
            for Polarity, polarity_int in sorted(polarity_to_id.items()):
                    features_chi2 = chi2(features, labels == polarity_int)
                    indices = np.argsort(features_chi2[0])
                    feature_names = np.array(tfidf.get_feature_names())[indices]
                    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
                    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]    

            from sklearn.model_selection import train_test_split
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.feature_extraction.text import TfidfTransformer
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.linear_model import LogisticRegression

            from sklearn.metrics import accuracy_score

            X = df['text']
            y = df['Polarity_int']
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            vectorizer = TfidfVectorizer(stop_words='english')
            X_train_dtm = vectorizer.fit_transform(X_train)
            X_test_dtm = vectorizer.transform(X_test)

            clf_lr = LogisticRegression()
            clf_lr.fit(X_train_dtm, y_train)
            y_pred = clf_lr.predict(X_test_dtm)
            lr_score = accuracy_score(y_test, y_pred) # perfectly balanced binary classes

            clf_mnb = MultinomialNB()
            clf_mnb.fit(X_train_dtm, y_train)
            y_pred = clf_mnb.predict(X_test_dtm)
            mnb_score = accuracy_score(y_test, y_pred) # perfectly balanced binary classes

            prediction=clf_lr.predict(vectorizer.transform([input_text]))
            print(prediction)

            print(prediction)

            if prediction == 0:
                prediction = 'Neutral'
            elif prediction == 1:
                prediction = 'Positive'
            else:
                prediction = 'Negative'
            

            df = pd.DataFrame(list(zip(prediction)),columns =['Sentiment'])
            
            # if df['Sentiment'] == '0':
            #     df['Sentiment'] = 'Neutral'
            # elif df['Sentiment'] == '1':
            #     df['Sentiment'] = 'Positive'
            # else:
            #     df['Sentiment'] = 'Negative'

            st.write(df)











