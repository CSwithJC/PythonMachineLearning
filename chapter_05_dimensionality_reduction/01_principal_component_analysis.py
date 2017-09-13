import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""Principal Component Analysis (PCA)

PCA is a statistical procedure that uses an orthogonal transformation
to convert a set of observations of possibly correlated variables into
a set of values of linearly uncorrelated variables called principal
components (or sometimes, principal modes of variation).
"""

# Steps for the PCA Algorithm:

# 1. Standardize the d-dimension dataset
df_wine = pd.read_csv('../data/wine.csv')

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=0)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

# 2. Construct the Covariance Matrix
cov_mat = np.cov(X_train_std.T)

# 3. Decompose covariance matrix into eigenvectors and eigenvalues
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalyes \n%s' % eigen_vals)

# Sidestep: plot the data
tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        label='individual explained variance')

plt.step(range(1, 14), cum_var_exp, where='mid',
         label='cumulative explained variance')

plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

# 4. Select k eigenvectors that correspond to the k largest eigenvalues,
#    where k is the dimensionality of the new feature subpsace (k <= d)

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)
X_train_pca = X_train_std.dot(w)

# Now, lets visualize the transformed Wine dataset:
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == 1, 0],
                X_train_pca[y_train == 1, 1],
                c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

