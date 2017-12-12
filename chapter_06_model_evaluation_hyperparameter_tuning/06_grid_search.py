import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


"""Grid Search:

Grid search is nothing more than a brute force exhaustive search paradigm
where we specify a list of values for different hyperparameters and the computer
evaluates the model performance for each combination of those to obtain
the optimal set.
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

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_train, y_train)

print('Using grid search, the best score was:')
print(gs.best_score_)
print('Using grid search, the best hyperparameters are:')
print(gs.best_params_)

# Finally, choose the best estimator and train with the optimal parameters:
clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))

# Remember that Grid Search is very computationally expensive; an
# alternative approach is to use RandomizedSearchCV.
