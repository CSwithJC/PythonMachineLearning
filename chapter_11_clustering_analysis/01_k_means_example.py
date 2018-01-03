import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

"""Plot blobs that will be used for K-Means later"""
X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

plt.scatter(X[:, 0],
            X[:, 1],
            c='red',
            marker='o',
            s=50)

plt.grid()
plt.show()
