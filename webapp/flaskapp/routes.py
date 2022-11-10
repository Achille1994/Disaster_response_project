from flaskapp import app
from files.process_tokenize import tokenize
import json
import plotly
import pandas as pd
import nltk
import re
import pickle
nltk.download(['punkt', 'wordnet', 'stopwords','omw-1.4'])
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import render_template, request, jsonify, Flask
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
import sqlite3 


# load data
engine = create_engine('sqlite:///../webapp/data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
global model

#update custom pickler
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'tokenize':
            from files.process_tokenize import tokenize
            return tokenize
        return super().find_class(module, name)

model = CustomUnpickler(open("../webapp/models/classifier.pkl", 'rb')).load()



# master webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/master')
def master():
    
    # extract data needed for visuals 
    genre_counts = df.groupby('genre').count()['message'].sort_values(ascending=False)
    genre_names = list(genre_counts.index)
    Reponse_names=df[df.columns[4:14]].sum().sort_values(ascending=False).index.tolist()
    Reponse_counts=df[df.columns[4:14]].sum().sort_values(ascending=False).values.tolist()
    
    
    # create visuals 1
    graphs1 = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # create visuals 2
    graphs2 = [
        {
            'data': [
                Bar(
                    x=Reponse_names,
                    y=Reponse_counts
                )
            ],

            'layout': {
                'title': 'Distribution of top 10 Responses',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Reponse"
                }
            }
        }
    ]
    # list of two graphs
    graphs=[graphs1,graphs2]
    # encode plotly graphs in JSON
    ids= ["graph-{}".format(i) for i, _ in enumerate(graphs)] 
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder) 
  
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
   

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )
   
    