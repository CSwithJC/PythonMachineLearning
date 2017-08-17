import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from chapter_02.adaline import AdalineGD
from chapter_02.plot_perceptron_decision_regions import plot_decision_regions

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

if plot_without_standarization:
    """Plot with learning rate 0.01"""
    ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')

    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    """Plot with learning rate 0.0001"""
    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')

    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(Sum-squared-error)')
    ax[1].set_title('Adaline - Learning rate 0.0001')

    plt.show()

"""Try again, but this time with Standarization"""

if plot_with_standarization:

    # Standarize the data:
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    ada = AdalineGD(n_iter=15, eta=0.01)
    ada.fit(X_std, y)

    plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standarized]')
    plt.ylabel('petal length [standarized]')
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.show()