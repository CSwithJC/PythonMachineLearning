import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

"""In the plot generated here we can see that the elbow
is at around # of clusters = 3, so we should probably use
3 clusters for kmeans"""

X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)

plt.plot(range(1, 11),
         distortions,
         marker='o')

plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
