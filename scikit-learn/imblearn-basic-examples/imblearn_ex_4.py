import pandas as pd
import numpy as np
import os
import time

from collections import Counter

from functools import wraps

from imblearn.keras import BalancedBatchGenerator

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, BatchNormalization

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import \
    FunctionTransformer, \
    OneHotEncoder, \
    StandardScaler

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def convert_float64(X):
    return X.astype(np.float64)


def make_model(n_features):
    model = Sequential()
    model.add(Dense(200, input_shape=(n_features,),
              kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, kernel_initializer='glorot_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(50, kernel_initializer='glorot_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(25, kernel_initializer='glorot_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def timeit(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start_time = time.time()
        result = f(*args, **kwds)
        elapsed_time = time.time() - start_time
        print('Elapsed computation time: {:.3f} secs'
              .format(elapsed_time))
        return (elapsed_time, result)
    return wrapper


@timeit
def fit_predict_imbalanced_model(X_train, y_train, X_test, y_test):
    model = make_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=2, verbose=1, batch_size=1000)
    y_pred = model.predict_proba(X_test, batch_size=1000)
    return roc_auc_score(y_test, y_pred)


@timeit
def fit_predict_balanced_model(X_train, y_train, X_test, y_test):
    model = make_model(X_train.shape[1])
    training_generator = BalancedBatchGenerator(X_train, y_train,
                                                batch_size=1000,
                                                random_state=42)
    model.fit_generator(generator=training_generator, epochs=5, verbose=1)
    y_pred = model.predict_proba(X_test, batch_size=1000)
    return roc_auc_score(y_test, y_pred)


if __name__ == "__main__":
    training_data = pd.read_csv('./input/train.csv')
    testing_data = pd.read_csv('./input/test.csv')

    y_train = training_data[['id', 'target']].set_index('id')
    X_train = training_data.drop('target', axis=1).set_index('id')
    X_test = testing_data.set_index('id')

    print('The data set is imbalanced: {}'.format(Counter(y_train['target'])))

    numerical_columns = [name for name in X_train.columns
                         if '_calc_' in name and '_bin' not in name]
    numerical_pipeline = make_pipeline(
        FunctionTransformer(func=convert_float64, validate=False),
        StandardScaler()
    )

    categorical_columns = [name for name in X_train.columns
                           if '_cat' in name]
    categorical_pipeline = make_pipeline(
        SimpleImputer(missing_values=-1, strategy='most_frequent'),
        OneHotEncoder(categories='auto')
    )

    preprocessor = ColumnTransformer(
        [('numerical_preprocessing', numerical_pipeline, numerical_columns),
         ('categorical_preprocessing', categorical_pipeline,
          categorical_columns)],
        remainder='drop'
    )

    skf = StratifiedKFold(n_splits=10)

    cv_results_imbalanced = []
    cv_time_imbalanced = []
    cv_results_balanced = []
    cv_time_balanced = []

    for train_idx, valid_idx in skf.split(X_train, y_train):
        X_local_train = preprocessor.fit_transform(X_train.iloc[train_idx])
        y_local_train = y_train.iloc[train_idx].values.ravel()
        X_local_test = preprocessor.fit_transform(X_test.iloc[valid_idx])
        y_local_test = y_train.iloc[valid_idx].values.ravel()

        elapsed_time, roc_auc = fit_predict_imbalanced_model(
            X_local_train, y_local_train, X_local_test, y_local_test)
        cv_time_imbalanced.append(elapsed_time)
        cv_results_imbalanced.append(roc_auc)

        elapsed_time, roc_auc = fit_predict_balanced_model(
            X_local_train, y_local_train, X_local_test, y_local_test)
        cv_time_balanced.append(elapsed_time)
        cv_results_balanced.append(roc_auc)