import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def preprocess_data(data):

    target = 'Survived'
    data = data.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])
    data_y = data[target].values
    data = data.drop(labels=[target], axis=1)

    test_size = 0.20
    train_x, test_x, train_y, test_y = train_test_split(data, data_y, test_size=test_size, random_state=1789)

    emb_val = train_x['Embarked'].mode().item()
    train_x['Embarked'] = train_x['Embarked'].fillna(value=emb_val)
    test_x['Embarked'] = test_x['Embarked'].fillna(value=emb_val)

    age_val = train_x['Age'].quantile(q=0.5).item()
    train_x['Age'] = train_x['Age'].fillna(value=age_val)
    test_x['Age'] = test_x['Age'].fillna(value=age_val)

    train_x = train_x.replace({'Sex': {'male': 0, 'female': 1}})
    train_x = train_x.replace({'Embarked': {'S': 0, 'C': 1, 'Q': 2}})
    test_x = test_x.replace({'Sex': {'male': 0, 'female': 1}})
    test_x = test_x.replace({'Embarked': {'S': 0, 'C': 1, 'Q': 2}})

    return train_x.values, train_y, test_x.values, test_y


def train_titanic():
    data_path = Path('datasets/titanic')
    data_raw = pd.read_csv(data_path / 'train.csv')
    train_x, train_y, test_x, test_y = preprocess_data(data_raw)

    np.save(data_path / 'train_x.npy', train_x)
    np.save(data_path / 'train_y.npy', train_y)
    np.save(data_path / 'test_x.npy', test_x)
    np.save(data_path / 'test_y.npy', test_y)

    random_state = 123
    clf = AdaBoostClassifier(random_state=random_state)
    clf.fit(train_x, train_y)

    print(model_eval(clf.predict, test_x, test_y))

    model_path = Path('models/titanic')
    os.makedirs(model_path, exist_ok=True)
    pickle.dump(clf, open(model_path / 'titanic.pkl', 'wb'))


def model_eval(model, data_x, data_y):
    preds = model(data_x)
    return classification_report(data_y, preds)


if __name__ == '__main__':
    train_titanic()
