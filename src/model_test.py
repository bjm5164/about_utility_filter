#!/usr/bin/python3
# model_run.py

import pandas as pd
import preprocess as pp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import cloudpickle as cpk

def test():

    test_data_fname = "../data/testing.csv"
    vectorizer_fname = "vectorizer.pickle"
    model_fname = 'about_classifier_LR.pickle'


    # Load, read and normalize data

    df = pd.read_csv(test_data_fname)
    df['Label_n'] = (pd.get_dummies(df.Label, drop_first=True, dtype=int) - 1) * -1
    df = pp.preprocess(df, lemmatize_text=True, remove_stops=True)
    
    #Load the vectorizer
    vectorizer = cpk.load(open(vectorizer_fname, "rb"))
    
    # Generate tfidf matirx
    X_test = df.TextClean
    y_test = df.Label_n
    _, features_test = pp.TFIDF_vectorize(X_test=X_test, vectorizer=vectorizer, return_vectorizer=False) 
    
    #Load the model
    clf_LR = cpk.load(open(model_fname, "rb"))
    
    #predict
    y_predicted = clf_LR.predict(features_test)
    
    
    #Model Evaluation metrics 
    print('Accuracy Score : ' + str(accuracy_score(y_test,y_predicted)))
    print('Precision Score : ' + str(precision_score(y_test,y_predicted)))
    print('Recall Score : ' + str(recall_score(y_test,y_predicted)))
    print('F1 Score : ' + str(f1_score(y_test,y_predicted)))
    

    
if __name__ == '__main__':
    test()