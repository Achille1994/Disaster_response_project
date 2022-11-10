import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from sklearn.externals import joblib
import pickle


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    # features variables
    X =df.message.values
    # targets  variables
    Y =df.drop(['id','original','genre',"message"],axis=1).values
    # category names variables
    category_names=df.drop(['id','original','genre',"message"],axis=1).columns
    return X, Y, category_names 


def tokenize(text):
      
    # Remove stop words
    stop_words = stopwords.words("english")
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # remove punctuation,lemmatize and tokenize
    
    tokens=word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    clean_tokens=[lemmatizer.lemmatize(word)
                  for word in tokens if word not in stop_words]
    
    return clean_tokens
    


def build_model():
    # Build a machine learning pipeline
    model_pipeline=Pipeline([
        ('tfidfvect', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # GridSearchCV is used to find the best parameters for the model
    parameters = {
        
       'clf__estimator__min_samples_split': [5,10]
    }

    model = GridSearchCV(model_pipeline, param_grid=parameters, cv=2)
    

    return model
   


def evaluate_model(model, X_test, Y_test, category_names):
    # predict testing
    Y_test_preds=model.predict(X_test)
    #  classification_reports
    for i in range(len(category_names)):
        print("category_names :", category_names[i],"\nclassification_reports:")
        testing_score=classification_report(Y_test[:,i],Y_test_preds[:,i])
        print(testing_score)


def save_model(model, model_filepath):
    # save the model to disk
    #joblib.dump(model,model_filepath )
    pickle.dump(model, open(str(model_filepath), 'wb'))
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()