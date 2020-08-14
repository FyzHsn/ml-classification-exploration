import csv
import datetime as dt
import numpy as np
import pandas as pd
import random

from scipy.sparse import csc_matrix, hstack 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler


NUMERICAL_FEAT = ['startCount', 'viewCount', 'clickCount', 'installCount',
                  'startCount1d', 'startCount7d']

CATEGORICAL_FEAT = ['campaignId', 'sourceGameId', 'country', 'platform',
                    'softwareVersion', 'connectionType', 'deviceType']


def load_data_as_df(file_path, sample_size=None):
    """Load dataset from memory in disk.

    :param file_path: data file path
    :type file_path: str
    :param sample_size: size of data subsample
    :type sample_size: int
    :return: subsample of dataset
    :rtype: pd.DataFrame
    """

    data = []
    with open(file_path) as f:
        reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_NONE)
        for row in reader:
            data.append(row)
    col_names = data.pop(0)

    if sample_size is not None:
        shuffled_indices = list(range(sample_size))
        random.shuffle(shuffled_indices)
        df = pd.DataFrame([data[idx] for idx in shuffled_indices],
                          columns=col_names)
    else:
        df = pd.DataFrame(data, columns=col_names)

    for feat in NUMERICAL_FEAT:
        df[feat] = df[feat].astype('int')
    return df


def datetime_parser(datetime_str):
    """Parse dataframe datetime object

    :param datetime_str: date and time information
    :type datetime_str: str
    :return: date and time information
    :rtype: datetime.datetime or None
    """

    try:
        date_str, time_str = datetime_str.split("T")
    except:
        return
    time_str = time_str[:8]
    year, month, day = date_str.split("-")
    hour, minut, sec = time_str.split(":")
    return dt.datetime(int(year), int(month), int(day), int(hour), int(minut),
                       int(sec))


def time_diff_in_minutes(dt_0, dt_1):
    """Find time difference

    :param dt_0: initial time
    :type dt_0: datetime.datetime
    :param dt_1: final time
    :type dt_1: datetime.datetime
    :return: time difference
    :rtype: float
    """

    return np.round((dt_1 - dt_0).total_seconds() / 60.0, 0)


def engineer_features(df):
    """Perform feature engineering

    :param df: dataset
    :type df: pd.DataFrame
    :return: dataset post feature engineering
    :rtype: pd.DataFrame
    """

    df.timestamp = df.timestamp.apply(datetime_parser)
    df.lastStart = df.lastStart.apply(datetime_parser)

    df['timeSinceLastStart'] = df.apply(
        lambda row: time_diff_in_minutes(row['lastStart'], row['timestamp']),
        axis=1).fillna(0)
    return df


def undersample(x, y):
    """Undersample a dataset

    :param x: feature vectors
    :type x: scipy.sparse.csc.csc_matrix
    :param y: target vector
    :type y: list
    :return: resampled feature and target vector
    :rtype: tuple of scipy.sparse.csc.csc_matrix and list
    """

    zero_indices = []
    one_indices = []
    for idx, class_label in enumerate(y):
        if class_label == 0:
            zero_indices.append(idx)
        else:
            one_indices.append(idx)

    resampled_indices = one_indices + random.sample(zero_indices,
                                                    len(one_indices))
    random.shuffle(resampled_indices)
    resampled_x, resampled_y = [], []
    for idx in resampled_indices:
        resampled_x.append(x[idx].toarray()[0])
        resampled_y.append(y[idx])
    return csc_matrix(resampled_x), resampled_y


if __name__ == "__main__":

    # Load training data
    df_train = load_data_as_df("training_data.csv", 800000)
    df_train['install'] = df_train['install'].astype('int')
    df_train.head()
    print('Training data loaded')


    # Load test data
    df_test = load_data_as_df("training_data.csv")
    df_test.head()
    print('Test data loaded')

    # Feature engineering
    df_train = engineer_features(df_train)
    df_test = engineer_features(df_test)
    print('Feature engineering completed')

    # One Hot Encoding Categorical Features Of Train/Test Sets
    X_train_num = df_train[NUMERICAL_FEAT].values
    X_train_cat = df_train[CATEGORICAL_FEAT]

    X_test_num = df_test[NUMERICAL_FEAT].values
    X_test_cat = df_test[CATEGORICAL_FEAT]

    enc = OneHotEncoder(handle_unknown='ignore')
    X_train_cat = enc.fit_transform(X_train_cat)
    X_train = csc_matrix(hstack((X_train_cat, X_train_num)))

    X_test_cat = enc.transform(X_test_cat)
    X_test = csc_matrix(hstack((X_test_cat, X_test_num)))

    y_train = df_train['install'].values
    print('One hot encoding successful')


    # Model Training
    X_train, y_train = undersample(X_train, y_train)
    print('Data undersampled')

    pipe_lr = Pipeline([('scl', MaxAbsScaler()),
                        ('clf', LogisticRegression(C=0.1, penalty='l1',
                                                   solver='liblinear',
                                                   random_state=0))])
    pipe_lr.fit(X_train, y_train)
    print('Training completed')

    # Generate Predictions
    y_proba = pipe_lr.predict_proba(X_test):
    df_test['install_proba'] = y_proba[:, 1]
    df_predictions = df_test[['id', 'install_proba']]
    df_predictions.to_csv('predictions.csv', index=False)

