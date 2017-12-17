import theano
import numpy as np
from theano import tensor as T

# Config Theano to use 32-bit architecture:
theano.config.floatX = 'float32'
#theano.config.device = 'gpu'

# initialize
data = np.array([[1, 2, 3]],
                dtype=theano.config.floatX)
x = T.fmatrix(name='x')
w = theano.shared(np.asarray([[0.0, 0.0, 0.0]],
                             dtype=theano.config.floatX))
z = x.dot(w.T)
update = [[w, w+1.0]]

# compile
net_input = theano.function(inputs=[x],
                            updates=update,  # Update variable
                            givens={x: data},  # Givens is used to insert values into a graph
                                               # before compiling it
                            outputs=z)

# execute
data = np.array([[1, 2, 3]])
for i in range(5):
    print('z%d:' % i, net_input(data))
