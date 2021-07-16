import pandas as pd
from sklearn.model_selection import train_test_split



if __name__ == '__main__':

    df = pd.read_csv('../data/Aboutlabeled.csv')
    X_train, X_test = train_test_split(df, random_state=42)
    X_train.to_csv('../data/training.csv')
    X_test.to_csv('../data/testing.csv')