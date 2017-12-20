import os
import struct
import numpy as np
import theano

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

"""Train a neural network using MNIST data and Keras
"""


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2

    return images, labels


# Load data and print the dimensions
X_train, y_train = load_mnist('../data/', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

X_test, y_test = load_mnist('../data/', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

theano.config.floatX = 'float32'

# Cast image data in 32-bit format:
X_train = X_train.astype(theano.config.floatX)
X_test = X_test.astype(theano.config.floatX)

print('First 3 labels: ', y_train[:3])

# Convert class labels into one-hot format:
y_train_ohe = np_utils.to_categorical(y_train)

print('\nFirst 3 labels (one-hot):\n', y_train_ohe[:3])

# Hyperbolic Tangent activation functions, softmax, and add an
# additional hidden layer.
np.random.seed(1)

# Initialize model with Sequential() to implement a feedforward neural network.
model = Sequential()

# First layer; input dimension must be the same as the training data.
model.add(Dense(input_dim=X_train.shape[1],
                output_dim=50,
                init='uniform',
                activation='tanh'))

# output_dim of previous layer and input_dim of this layer must match
model.add(Dense(input_dim=50,
                output_dim=50,
                init='uniform',
                activation='tanh'))

# Number of units in the output layer should be equal to the number
# of unique class labels
model.add(Dense(input_dim=50,
                output_dim=y_train_ohe.shape[1],
                init='uniform',
                activation='softmax'))

# Stochastic Gradient Descent Optimizer
sgd = SGD(lr=0.001, decay=1e-7, momentum=0.9)

model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train,
          y_train_ohe,
          nb_epoch=50,
          batch_size=300,
          verbose=1,
          validation_split=0.1)

y_train_pred = model.predict_classes(X_train, verbose=0)
print('First 3 Predictions: ', y_train_pred[:3])

train_acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (train_acc * 100))

y_test_pred = model.predict_classes(X_test, verbose=0)
print('First 3 Predictions: ', y_test_pred[:3])

test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Testing accuracy: %.2f%%' % (test_acc * 100))
