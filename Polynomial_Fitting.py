import tensorflow as tf
import numpy as np
import random

x_data = np.float32(np.random.rand(100)) # Random input
y_data = np.sin(2*np.pi*x_data) + 0.3 * np.random.rand()

W3 = tf.Variable(random.random())
W2 = tf.Variable(random.random())
W1 = tf.Variable(random.random())
W0 = tf.Variable(random.random())

y = W3*x_data*x_data*x_data+W2*x_data*x_data + W1*x_data + W0

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5) #最急降下法
train = optimizer.minimize(loss)

# For initializing the variables.
init = tf.initialize_all_variables()

# Launch the graph
sess = tf.Session()
sess.run(init)

# Fit the plane.
for step in range(5000):
    sess.run(train)
print(step, sess.run(W3), sess.run(W2), sess.run(W1), sess.run(W0))
