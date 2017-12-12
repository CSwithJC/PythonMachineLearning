import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, make_scorer


"""Precision, Recall, F1-Score:

Precision (also called positive predictive value) is the fraction of
relevant instances among the retrieved instances, while recall
(also known as sensitivity) is the fraction of relevant instances
that have been retrieved over the total amount of relevant instances.
F1-Score is calculated using both precision and recall.

"""
df = pd.read_csv('../data/wdbc.csv')

# Convert M (malign) and B (benign) into numbers
X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values

le = LabelEncoder()
y = le.fit_transform(y)

le.transform(['M', 'B'])

# Divide Dataset into separate training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

pipe_svc = Pipeline([('scl', StandardScaler()),
                     ('clf', SVC(random_state=1))])

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)

# Create the Confusion Matrix:
confmat = confusion_matrix(y_true=y_test,
                           y_pred=y_pred)
print(confmat)

# Precision, Recall, F1-Score:
print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

# We can do grid search based on any of these scores, not just accuracy:
# All the different values we will try for C.
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# Combinations of different parameters to be ran here for Grid Search:
param_grid = [
    {
        'clf__C': param_range,
        'clf__kernel': ['linear']
    },
    {   # NOTE: gamma parameter is specific to kernel SVMs.
        'clf__C': param_range,
        'clf__gamma': param_range,
        'clf__kernel': ['rbf']
    }
]

scorer = make_scorer(f1_score, pos_label=0)
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring=scorer,
                  cv=10,
                  n_jobs=-1)
