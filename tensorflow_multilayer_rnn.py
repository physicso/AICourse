"""
To classify images using a recurrent neural network, we consider every image row as a sequence of pixels.
Because MNIST image shape is 28*28px, we will then handle 28 sequences of 28 steps for every sample.

This script works on TensorFlow 1.x :)
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

print('The shape of input images:')
print(mnist.train.images.shape)

# Basic Parameters
learning_rate = 0.0001
training_iters = 10000
batch_size = 100
display_step = 100

# Network Parameters
n_input = 28 # MNIST data input (Image shape: 28*28)
n_steps = 28 # Timesteps
n_hidden = 128 # Hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)
n_layers = 2 #LSTM layer num
# LSTM_CELL Definition
lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=0.0, state_is_tuple=True)
cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * n_layers, state_is_tuple=True)
_state = cell.zero_state(batch_size, tf.float32) # Tensorflow LSTM cell requires 2 x n_hidden length (state & cell)
# Define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
# Create Graph
x = tf.placeholder("float32", [None, n_steps, n_input])
y = tf.placeholder("float32", [None, n_classes])
# Transform the input into ones for RNN
a1 = tf.transpose(x, [1, 0, 2])
a2 = tf.reshape(a1, [-1, n_input])
a3 = tf.matmul(a2, weights['hidden']) + biases['hidden']
a4 = tf.split(a3, n_steps, 0) # NOTE: a4 = tf.split(0, n_steps, a3) for TF version < 1.12
# RNN Construction
outputs, states = tf.contrib.rnn.static_rnn(cell, a4, initial_state = _state)
pred = tf.matmul(outputs[-1], weights['out']) + biases['out'] #outputs[-1] is the output of the last timestamp
# Define Cost, Accuracy, and Optimization Method
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) #softmax_cross_entropy_with_logits avoids log0!
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Let's Do It!
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)


step = 1
while step < training_iters:
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    # Reshape data to get 28 seq of 28 elements
    batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
    # Fit training using batch data
    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
    if step % display_step == 0:
            # Calculate batch accuracy
        acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,})
            # Calculate batch loss
        loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
        print('Iteration: %5d | Minibatch loss: %.6f | Training accuracy: %.6f' %
              (step, loss, acc))
    step += 1

test_len = batch_size
test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
test_label = mnist.test.labels[:test_len]
# Evaluate model
print('Test accuracy: %.6f' % sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

'''
Program Output:

Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
The shape of input images:
(55000, 784)
Iteration:   100 | Minibatch loss: 1.486994 | Training accuracy: 0.510000
Iteration:   200 | Minibatch loss: 0.812069 | Training accuracy: 0.770000
...
Test accuracy: 0.993663
'''
