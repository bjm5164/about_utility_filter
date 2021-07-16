#!/usr/bin/python3
# model_train.py


import preprocess as pp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import cloudpickle as cpk



def train():

    # Load, read and vectorize training data
    train_data_fname = "../data/training.csv"
    vectorizer_fname = "vectorizer.pickle"
    model_fname = 'about_classifier_LR.pickle'
    
    df = pd.read_csv(train_data_fname)
    df['Label_n'] = (pd.get_dummies(df.Label, drop_first=True, dtype=int) - 1) * -1
    df = pp.preprocess(df, lemmatize_text=True, remove_stops=True)
    
    # Generate tfidf matirx
    X_train = df.TextClean
    y_train = df.Label_n
    features_train, _, vectorizer = pp.TFIDF_vectorize(X_train, vectorizer=None) 

    #Fix class imbalance
    smote = SMOTE(random_state = 101)
    X_bal, y_bal = smote.fit_resample(features_train, y_train)
    
    #Train model
    clf_LR = LogisticRegression(max_iter=1000, solver='saga', penalty='l2', C=25)
    clf_LR.fit(X_bal,y_bal.ravel())
    
    #Save model and vectorizer
    cpk.dump(clf_LR, open(model_fname, 'wb'))
    cpk.dump(vectorizer, open(vectorizer_fname, "wb"))
        
   
        
if __name__ == '__main__':
    train()