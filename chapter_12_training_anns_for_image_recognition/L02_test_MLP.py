import os
import struct

import matplotlib.pyplot as plt
import numpy as np

from chapter_12_training_anns_for_image_recognition.neural_net import NeuralNetMLP


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

X_train, y_train = load_mnist('../data/', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

X_test, y_test = load_mnist('../data/', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

nn = NeuralNetMLP(n_output=10,  # 10 Output units (because numbers 0 to 9)

                  n_features=X_train.shape[1],  # 784 Input Features

                  n_hidden=50,  # 50 Hidden Units

                  l2=0.1,  # L2 Regularization to decrease degree of overfitting

                  l1=0.0,  # L1 Regularization

                  epochs=1000,  # 1000 passes over the training set

                  eta=0.001,  # Learning Rate

                  alpha=0.001,  # Parameter for Momentum Learning

                  decrease_const=0.00001,  # Decrease constant d for an adaptive learning rate
                                           # that decreases over time for better convergence

                  shuffle=True,  # Shuffle training set prior to ever epoch to prevent
                                 # getting stuck in a cycle

                  minibatches=50,  # Splitting of the training data into k mini-batches in each
                                   # epoch; gradient is computed for each mini-batch separately
                                   # instead of the entire training data for faster learning.

                  random_state=1)

# Train Neural Network using 60,000 samples from MNIST training dataset.
nn.fit(X_train, y_train, print_progress=True)

# Plot the cost for each epoch
plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epocks * 50 ')
plt.tight_layout()
plt.show()

# Plot Smoother version of the cost function by averaging over mini-batch intervals
batches = np.array_split(range(len(nn.cost_)), 1000)
cost_ary = np.array(nn.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]

plt.plot(range(len(cost_avgs)),
         cost_avgs,
         color='red')
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()

# Evaluate performance by calculating prediction accuracy:

# Train Data
y_train_pred = nn.predict(X_train)
acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('\nTraining Accuracy: %.2f%%' % (acc * 100))

# Test Data
y_test_pred = nn.predict(X_test)
acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('\nTest Accuracy: %.2f%%' % (acc * 100))

# Plot some of the images the MLP struggled with:
miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]

fig, ax = plt.subplots(nrows=5,
                       ncols=5,
                       sharex=True,
                       sharey=True,)

ax = ax.flatten()

for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img,
                 cmap='Greys',
                 interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
