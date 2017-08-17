import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from chapter_02_linear_classifiers.adaline_stochastic import AdalineSGD
from chapter_02_linear_classifiers.plot_perceptron_decision_regions import plot_decision_regions

plot_without_standarization = False
plot_with_standarization = True

df = pd.read_csv('../data/iris.csv', header=None)

# store the values of the labels for the first 100 training samples
y = df.iloc[0:100, 4].values

# make Iris-setosa -1 and Iris-versicolor into 1
y = np.where(y == 'Iris-setosa', -1, 1)

# store the first (sepal length) and third (petal length) feature columns
# for the first 100 training samples
X = df.iloc[0:100, [0, 2]].values

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

# Standarize the data:
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada)

plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standarized]')
plt.ylabel('petal length [standarized]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()