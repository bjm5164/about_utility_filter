import pandas as pd
import numpy as np
import re
import sys
import nltk
import contractions
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


# Define methods

def tokenize(corp):
    '''tokenize a string
    
    args:
    corp: str, text corpus
    
    returns:
    list, tokenized string
    '''
    # expand contractions
    expanded = [contractions.fix(word) for word in corp.split()]
    corp = ' '.join(expanded)
    
    # tokenize
    tk = TweetTokenizer(preserve_case=False)
    split = tk.tokenize(corp) 
    
    # drop non-alpha numerics
    alphanum = [re.findall('[a-zA-Z0-9]+',i) for i in split]
    appended = [''.join(i) for i in alphanum]
    
    return list(filter(None, appended))


def get_wordnet_pos(tag):
    ''' Convert nltk wordnet parts of speech to nltk lemmatizer parts of speech. This could use expansion.
    
    args:
    tag: str, part of speech output from .pos_tag method
    
    returns:
    pos: wordnet part of speech for nltk lemmatizer'''
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V') or tag.startswith('M'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    

def lemmatize(string, return_pos=True, lemmatizer=WordNetLemmatizer()):
    ''' Lemmatization step
    args:
    string: list, tokenized string
    return_pos: bool, return part of speech classification from nltk.pos_tag()
    lemmatizer: lemmatizer to use. Default is WordNetLemmatizer()
    
    returns: list, lemmatized tokens'''

      
    pos = nltk.pos_tag(string)
    if return_pos is True:    
        return [(lemmatizer.lemmatize(i[0], get_wordnet_pos(i[1])),i[1]) for i in pos]
    else:
        return [lemmatizer.lemmatize(i[0], get_wordnet_pos(i[1])) for i in pos]
    

def preprocess(df, 
               lemmatize_text=True, 
               remove_stops=True):
    
    '''Preprocess dataframes containing about strings.
    args:
    df: pandas.DataFrame, dataframe containing at least a ['Text'] column
    lemmatize_text: bool, run lemmatization on tokens, default=True
    remove_stops: bool, remove english stop words from tokens, default = True
    
    returns:
    df: pandas.DataFrame with new column ['TextClean'] and ['TextTokenized'] and (if lemmatize_text is True) ['TextLemmatized']
    '''
    #Tokenize
    df['TextTokenized'] = [tokenize(i) for i in df.Text]
    
    if lemmatize_text:
    #Lemmatize
        df['TextLemmatized'] = [lemmatize(i, return_pos=False) for i in df.TextTokenized]
    
    if remove_stops:  
    #Remove Stops
        stops = set(stopwords.words('english'))  
        if lemmatize_text:
            df['TextClean'] = [[i for i in entry if i not in stops] for entry in df.TextLemmatized]
        else:
            df['TextClean'] = [[i for i in entry if i not in stops] for entry in df.TextTokenized] 
    return df


def TFIDF_vectorize(X_train=None, X_test=None, vectorizer=None, return_vectorizer=True):
    '''Generate a TFIDF matrix from tokenized strings. This can be used to fit train and test data simultaneously, or train data only.
    args:
    X_train: iterable, iterable of tokenized strings used for training. 
    X_test: iterable, iterable of tokenized strings used for testing. Default=None 
    vectorizer: TfidfVectorizer or None, Pre-fit TfidfVectorizer or None.
    return_vectorizer: bool, return the fit vectorizer object for use later, Default=True
    '''
    
    if vectorizer is None:
        tk = lambda x: x
        vectorizer = TfidfVectorizer(tokenizer=tk, binary=True, stop_words=None, use_idf=True, lowercase=False, ngram_range=(1,2))
    
    
    if X_train is None:
        features_train = None 
    else:
        features_train = vectorizer.fit_transform(X_train)

    if X_test is None:
        features_test = None
    else:
        features_test = vectorizer.transform(X_test)
        
    
    if return_vectorizer is True:
        return features_train, features_test, vectorizer
    else:
        return features_train, features_test