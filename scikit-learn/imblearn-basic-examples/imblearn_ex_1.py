from collections import Counter

from imblearn.datasets import make_imbalance
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


RANDOM_STATE = 0

iris = load_iris()
X, y = make_imbalance(iris.data, iris.target,
                      sampling_strategy={0: 25, 1:50, 2: 50},
                      random_state=RANDOM_STATE)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=RANDOM_STATE)

print('Training target statistics: {}'.format(Counter(y_train)))
print('Test target statistics: {}'.format(Counter(y_test)))

pipeline = make_pipeline(NearMiss(version=2),
                         LinearSVC(random_state=RANDOM_STATE))
pipeline.fit(X_train, y_train)

print(classification_report_imbalanced(y_test, pipeline.predict(X_test)))
