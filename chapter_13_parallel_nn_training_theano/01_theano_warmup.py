import theano
from theano import tensor as T

# Initialize
x1 = T.scalar()
w1 = T.scalar()
w0 = T.scalar()
z1 = w1 * x1 + w0

# Compile
net_input = theano.function(inputs=[w1, x1, w0], outputs=z1)

# Execute
print('Net input: %.2f' % net_input(2.0, 1.0, 0.5))
