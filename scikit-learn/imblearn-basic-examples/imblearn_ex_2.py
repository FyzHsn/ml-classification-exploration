from collections import Counter

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced


categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics',
              'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train',
                                      categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',
                                     categories=categories)

X_train = newsgroups_train.data
X_test = newsgroups_test.data

y_train = newsgroups_train.target
y_test = newsgroups_test.target

print('Training class distributions summary: {}'.format(Counter(y_train)))
print('Test class distributions summary: {}'.format(Counter(y_test)))

# Regular scikit-learn pipeline
pipe = make_pipeline(TfidfVectorizer(), MultinomialNB())
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report_imbalanced(y_test, y_pred))

# Under sampled data pipeline
pipe = make_pipeline_imb(TfidfVectorizer(),
                         RandomUnderSampler(),
                         MultinomialNB())
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report_imbalanced(y_test, y_pred))