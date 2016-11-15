import cPickle as pickle
import gzip
from sklearn.neural_network import MLPClassifier


def read_data(data_file):
    f = gzip.open(data_file, "rb")
    train, val, test = pickle.load(f)
    f.close()
    train_x = train[0]
    train_y = train[1]
    test_x = test[0]
    test_y = test[1]
    return train_x, train_y, test_x, test_y


data_file = "mnist.pkl.gz"
print 'Reading training and testing data...'
X_train, y_train, X_test, y_test = read_data(data_file)
mlp = MLPClassifier(hidden_layer_sizes=(200,200), activation='relu', max_iter=100, alpha=1e-4,
                    solver='sgd', tol=1e-4, verbose=10, learning_rate_init=.1)
mlp.fit(X_train, y_train)
print("Training Set Accuracy: %f" % mlp.score(X_train, y_train))
print("Test Set Accuracy: %f" % mlp.score(X_test, y_test))