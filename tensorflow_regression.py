"""
This script works on TensorFlow 1.x :)
"""
import tensorflow as tf
import numpy


train_X = numpy.asarray([30.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0])
train_Y = numpy.asarray([320.0, 360.0, 400.0, 455.0, 490.0, 546.0, 580.0])
train_X /= 100.0
train_Y /= 100.0

X = tf.placeholder('float')
y = tf.placeholder('float')
W = tf.Variable(numpy.random.randn(), name='weight')
b = tf.Variable(numpy.random.randn(), name='bias')
n_samples = train_X.shape[0]
y_pred = tf.add(tf.multiply(X, W), b)


cost = tf.reduce_sum(tf.pow((y_pred - y), 2))/(2 * n_samples)


learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


training_epochs = 1000
display_step = 50

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for epoch in range(training_epochs):
        for (x_train, y_train) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x_train, y: y_train})
        if epoch % display_step == 0:
            print('Iteration: %04d | Loss: %.6f | W: %.6f | b: %.6f'
                  % (epoch + 1, sess.run(cost, feed_dict={X: train_X, y: train_Y}),
                     sess.run(W), sess.run(b)))
    training_cost = sess.run(cost, feed_dict={X: train_X, y: train_Y})
    print('Training loss: %.6f | W: %.6f | b: %.6f' % (training_cost, sess.run(W), sess.run(b)))
