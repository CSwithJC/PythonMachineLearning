import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh

from sklearn.datasets import make_moons


""" Kernel PCA

Same as before, but modify the function so that now it also returns eigenvalues
of the kernel matrix.
"""


def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation

    Parameters
    ----------
    X: {NumPy ndarray}, shape = [n_samples, n_features]

    gamma: float
        Tuning parameter of the RBF kernel

    n_components:
        Number of principal components to return

    Returns
    -------
    X_pc: {NumPy ndarray}, shape = [n_samples, n_features]
        Projected dataset

    lambdas: list
       Eigenvalues
    """

    # Calculate the pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # numpy.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)

    # Collect the top k eigenvectors (projected samples)
    alphas = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))

    # Collect the corresponding eigenvalues
    lambdas = [eigvals[-i] for i in range(1, n_components+1)]

    return alphas, lambdas

# Create half-moon dataset and project it onto a one-dimensional
# subspace using the updated RBF Kernel PCA implementation:
X, y = make_moons(n_samples=100, random_state=123)
alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)

# Let's assume the 26th point from the half-moon dataset
# is the new data point x', and our task is to project it
# onto this new subspace:

x_new = X[25]
print(x_new)

x_proj = alphas[25]  # original projection
print(x_proj)


def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)

x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
print(x_reproj)

plt.scatter(alphas[y == 0, 0], np.zeros((50)),
            color='red', marker='^', alpha=0.5)

plt.scatter(alphas[y == 1, 0], np.zeros((50)),
            color='blue', marker='o', alpha=0.5)

plt.scatter(x_proj, 0, color='black',
            label='original projection of point X[25]',
            marker='^', s=100)

plt.scatter(x_reproj, 0, color='green',
            label='remapped point X[25]',
            marker='x', s=100)

plt.legend(scatterpoints=1)
plt.show()

# As seen here, we mapped the sample x' onto the first
# principal component correctly.
