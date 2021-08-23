
from flask import Flask
from flask import request, make_response
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load


app = Flask(__name__)


df = pd.read_csv('../data/movies_with_labels.csv')

kmeans = load('../Models/kmeans-films.joblib')

count_vectorizer = TfidfVectorizer(stop_words='english')
count_vectorizer.fit_transform(df['documents'])

def get_label(text):
    
    y = count_vectorizer.transform([text])
    
    label_pred = kmeans.predict(y)
    
    return label_pred

def get_df_films(label, film):

    return df[((df['labels'] == label) | (df['title'] == film))].copy().drop_duplicates(subset=['title'], keep='last').reset_index(drop=True)

    
def get_recommendation(dataframe, film):
    
    indice_movies = pd.Series(dataframe.index, index=dataframe['title'])

    count_v = TfidfVectorizer(stop_words='english')
    matrix_tfidf = count_v.fit_transform(dataframe['description'])

    cos_sim = cosine_similarity(matrix_tfidf, matrix_tfidf)

    scores = sorted(list(enumerate(cos_sim[indice_movies[film]])), key=lambda l: l[1], reverse=True)

    movie_indices = [f[0] for f in scores[:10]]

    return dataframe['title'].iloc[movie_indices]


@app.route('/webhook', methods=['POST'])
def webhook():

    try:
        # Obtenemos la petición
        req = request.get_json(silent=True, force=True)

        query = req['queryResult']

        parameters = query['parameters']
        genre = parameters['Genre']
        film = parameters['movie']
        famous = parameters['famous']
        epoch = parameters['epoch']

        label = get_label(' '.join(genre) + ' '.join(famous) + ' '.join(epoch) + 'sci-fi')

        df_films = get_df_films(label[0], film)

        recommendations = get_recommendation(df_films, film)

        recommendations = ',\n'.join(recommendations.reset_index()['title'].tolist())

        return {
            'fulfillmentText': recommendations,
            'displayText': 25,
            'source': 'webhookdata'
        }
    except:
        return {
            'fulfillmentText': 'Give me another film please',
            'displayText': 25,
            'source': 'webhookdata'
        }
        
