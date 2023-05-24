import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def preprocess_data(data):
    data = data.drop(labels=['capital.gain', 'capital.loss', 'workclass', 'education', 'occupation', 'relationship', 'race', 'native.country'], axis=1)

    # print(f'Number of missing values: {data.isnull().sum().sum()}')

    data = data.replace({'income': {'<=50K': 0, '>50K': 1,
                                    '<=50K.': 0, '>50K.': 1}})

    data = data.replace({'sex': {'Male': 0, 'Female': 1},
                         'marital.status': {'Never-married': 0, 'Divorced': 0, 'Separated': 0, 'Widowed': 0,
                                            'Married-civ-spouse': 1, 'Married-spouse-absent': 1, 'Married-AF-spouse': 1}})

    data_y = data['income'].values
    data = data.drop(labels=['income'], axis=1)

    data_x = data.values

    test_size = 0.20
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=test_size, random_state=1789)

    return train_x, train_y, test_x, test_y


def train_census():
    data_path = Path('datasets/census')
    data_raw = pd.read_csv(data_path / 'adult.csv')
    train_x, train_y, test_x, test_y = preprocess_data(data_raw)

    np.save(data_path / 'train_x.npy', train_x)
    np.save(data_path / 'train_y.npy', train_y)
    np.save(data_path / 'test_x.npy', test_x)
    np.save(data_path / 'test_y.npy', test_y)

    rf = RandomForestClassifier(n_estimators=250, random_state=123)
    rf.fit(train_x, train_y)

    print(model_eval(rf.predict, test_x, test_y))

    model_path = Path('models/census')
    os.makedirs(model_path, exist_ok=True)
    pickle.dump(rf, open(model_path / 'census.pkl', 'wb'))


def model_eval(model, data_x, data_y):
    preds = model(data_x)
    return classification_report(data_y, preds)


if __name__ == '__main__':
    train_census()
