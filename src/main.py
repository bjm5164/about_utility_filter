import pandas as pd
import preprocess as pp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cloudpickle as cpk

def inference():

    input_data_fname = "../data/testing.csv"
    vectorizer_fname = "vectorizer.pickle"
    model_fname = 'about_classifier_LR.pickle'


    # Load, read and normalize data

    df = pd.read_csv(input_data_fname)
    df = pp.preprocess(df, lemmatize_text=True, remove_stops=True)
    
    #Load the vectorizer
    vectorizer = cpk.load(open(vectorizer_fname, "rb"))
    
    # Generate tfidf matirx
    X_test = df.TextClean
    _, features_test = pp.TFIDF_vectorize(X_test=X_test, vectorizer=vectorizer, return_vectorizer=False) 
    
    #Load the model
    clf_LR = cpk.load(open(model_fname, "rb"))
    
    #predict
    y_predicted = clf_LR.predict(features_test)
    
    #save outputs
    categorical = {1:'About', 0:'None'}
    df['Predicted_Label'] = [categorical[i] for i in y_predicted]
    df.to_csv('../data/output.csv')
    

    
if __name__ == '__main__':
    inference()