import theano
import numpy as np
from theano import tensor as T

# Config Theano to use 32-bit architecture:
theano.config.floatX = 'float32'
#theano.config.device = 'gpu'

# initialize
x = T.fmatrix(name='x')
x_sum = T.sum(x, axis=0)

# compile
calc_sum = theano.function(inputs=[x], outputs=x_sum)

# execute (Python List)
ary = [[1, 2, 3], [1, 2, 3]]
print('Column sum:', calc_sum(ary))

# execute (NumPy array)
ary = np.array([[1, 2, 3], [1, 2, 3]],
               dtype=theano.config.floatX)
print('Column sum:', calc_sum(ary))

print('TensorType: ', x.type())
