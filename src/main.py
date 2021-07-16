import pandas as pd
import preprocess as pp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cloudpickle as cpk
from fastapi import FastAPI

app = FastAPI()


@app.get('/')
def get_root():

	return {'message': 'About page filter model'}

@app.get('/tasks/train/{message}')
async def train_model(message: str):
    if 'train_model' in message:
        return train()
    else:
        return 'unrecognized'

@app.get('/tasks/test/{message}')
async def test_model(message: str):
    if 'test_model' in message:
        return test()
    else:
        return 'unrecognized'

@app.get('/tasks/infer/{message}')
async def infer(message):
    return inference(message)


def inference(message):
    
    vectorizer_fname = "vectorizer.pickle"
    model_fname = 'about_classifier_LR.pickle'


    # Load, read and normalize data
    df = pd.DataFrame(columns=['Text','TextClean'])
    df.loc[0,'Text'] = message
    df = pp.preprocess(df, lemmatize_text=True, remove_stops=True)
    
    #Load the vectorizer
    vectorizer = cpk.load(open(vectorizer_fname, "rb"))
    
    # Generate tfidf matirx
    payload = df.TextClean
    _, features_test = pp.TFIDF_vectorize(X_test=payload, vectorizer=vectorizer, return_vectorizer=False) 
    
    #Load the model
    clf_LR = cpk.load(open(model_fname, "rb"))
    
     #predict
    y_predicted = clf_LR.predict(features_test)
    y_probability = clf_LR.predict_proba(features_test)
    
    #save outputs
    categorical = {1:'About', 0:'None'}

    return {'label':categorical[y_predicted[0]], 'probability':y_probability[0,1]}
    
   
    

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
    
    return 'Training Complete'
    

    
