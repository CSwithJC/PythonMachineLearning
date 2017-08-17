import numpy as np


class Perceptron(object):

    """ Perceptron Classifier:

    Parameters
    ----------

    eta: float
        Learning rate (between 0.0 and 1.0)

    n_iter: int
        Passes over the training dataset

    Attributes
    ----------

    w_ : 1d-array
        Weights after fitting

    errors_ : list
        Number of misclassifications in every epoch
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ Fit Training Data

        :param X:
        :param y:
        :return:

        """

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        print('About to begin iterations for training Perceptron...\n\n')
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                print('iteration #', _)
                print('xi is ', xi)
                print('target is ', target)

                update = self.eta * (target - self.predict(xi))
                print('update is ', update)

                self.w_[1:] += update * xi
                print('self.w_[1:] is ', self.w_[1:])
                self.w_[0] += update
                print('self.w_[0] is ', self.w_[0])

                print('weights after updates is ', self.w_)

                errors += int(update != 0.0)
                print('errors for this iterations are ', errors)
                print('\n\n')

            self.errors_.append(errors)

        print('total errors are ', self.errors_)

        return self

    def net_input(self, X):
        """ Calculate net input """
        """ (just for calculating w^T * x """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """ Return the class label after unit step """
        return np.where(self.net_input(X) >= 0.0, 1, -1)