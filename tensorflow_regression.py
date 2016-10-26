import tensorflow as tf
import numpy


X = tf.placeholder("float")
y = tf.placeholder("float")
W = tf.Variable(numpy.random.randn(), name="weight")
b = tf.Variable(numpy.random.randn(),name="bias")


y_pred = tf.add(tf.mul(X, W), b)


cost = tf.reduce_sum(tf.pow(y_pred- y,2))/(2 * n_samples)


learning_rate =0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

​
train_X = numpy.asarray([30.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0])
train_Y =numpy.asarray([320.0, 360.0, 400.0, 455.0, 490.0, 546.0, 580])


training_epochs = 1000
display_step = 50
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for(x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        if epoch % display_step ==0:
            print"Epoch:" + '%04d'% (epoch+1) + "cost=" + "{:.9f}".format(sess.run(cost, feed_dict={X: train_X, Y:train_Y})) + "W=" + sess.run(W),"b=", sess.run(b)
    print "Optimization Finished!"
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print "Training cost = " + training_cost + "W = " + sess.run(W) + "b=" + sess.run(b) + '\n'
