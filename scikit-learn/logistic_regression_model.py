import numpy as np
import pandas as pd
import random

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def get_class_label_indices(df, target_var):
    zeroes_indices = df.index[df[target_var] == 0].tolist()
    ones_indices = df.index[df[target_var] == 1].tolist()
    return zeroes_indices, ones_indices


def upsampled_stratified_test_train_indices(zeros, ones, test_split=0.3):
    random.shuffle(zeros)
    random.shuffle(ones)

    m, n = int(test_split * len(zeros)), int(test_split * len(ones))

    train_indices = zeros[m:] + random.choices(ones[n:], k=len(zeros) - m)
    test_indices = zeros[:m] + ones[:n]

    random.shuffle(train_indices)
    random.shuffle(test_indices)

    return train_indices, test_indices


def generate_feature_and_target_vectors(df, indices):
    X = df.loc[indices, ['startCount', 'viewCount', 'clickCount',
                         'installCount']].values.tolist()
    y = np.ravel(df.loc[indices, ['install']].values)
    return X, y


if __name__ == "__main__":
    data = pd.read_csv('training_data.csv', sep=';')
    features = ['startCount', 'viewCount', 'clickCount', 'installCount',
                'lastStart', 'startCount1d', 'startCount7d', 'install']
    data = data[features]
    print(data.head())

    zeros, ones = get_class_label_indices(data, 'install')
    train, test = upsampled_stratified_test_train_indices(zeros, ones, 0.3)

    X_train, y_train = generate_feature_and_target_vectors(data, train)
    X_test, y_test = generate_feature_and_target_vectors(data, test)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.fit_transform(X_test)

    clf = LogisticRegression()
    clf.fit(X_train_std, y_train)
    y_pred = clf.predict(X_test_std)
    print(accuracy_score(y_test, y_pred))
    print(y_pred[:100], y_test[:100])
    print(confusion_matrix(y_pred, y_test))
