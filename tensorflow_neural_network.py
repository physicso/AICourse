from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W_fc1 = weight_variable([784, 200])
b_fc1 = bias_variable([200])
W_fc2 = weight_variable([200, 200])
b_fc2 = bias_variable([200])
W_out = weight_variable([200, 10])
b_out = bias_variable([10])
hidden_1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
hidden_2 = tf.nn.relu(tf.matmul(hidden_1, W_fc2) + b_fc2)
y = tf.nn.softmax(tf.matmul(hidden_2, W_out) + b_out)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

# Train
training_iteration = 10000
batch_size = 100
display_step = 50
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	for iter in range(training_iteration):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		train_step.run({x: batch_xs, y_: batch_ys})
		if iter % display_step == 0:
			print "Epoch:", '%04d' % (iter + 1), "accuracy =", "{:.9f}".format(
				sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
	print "Test Accuracy: " + "{:.9f}".format(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
