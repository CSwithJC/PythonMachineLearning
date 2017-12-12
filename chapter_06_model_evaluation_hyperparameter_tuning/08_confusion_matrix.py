import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix


"""Confusion Matrix:

A matrix that lays out the performance of a learning algorithm.
Its just a square matrix that reports the counts of the true positives,
true negatives, false positives, and false negatives of a classifier.
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

# Plot the Confusion Matrix:
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)

for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i,
                s=confmat[i, j],
                va='center', ha='center')

plt.xlabel('predicted value')
plt.ylabel('true label')
plt.show()
