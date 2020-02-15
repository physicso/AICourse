import time
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import accuracy_score
import cPickle as pickle
import gzip
 

# Logistic Regression Classifier: One vs. All
def logistic_regression_classifier(train_x, train_y):
    model = linear_model.LogisticRegression(solver='sag', max_iter=10000, penalty='l2')
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier: Softmax
def logistic_regression_classifier_multi(train_x, train_y):
    model = linear_model.LogisticRegression(multi_class='multinomial', max_iter=10000, solver='sag', penalty='l2')
    model.fit(train_x, train_y)
    return model
 
 
# SVM Linear Kernel Classifier
def svm_classifier(train_x, train_y):
    model = svm.LinearSVC()
    model.fit(train_x, train_y)
    return model


# SVM Polynomial Kernel Classifier
def svm_poly_classifier(train_x, train_y):
    model = svm.SVC(kernel='poly')
    model.fit(train_x, train_y)
    return model
 

def read_data(data_file):
    f = gzip.open(data_file, "rb")
    train, val, test = pickle.load(f)
    f.close()
    train_x = train[0]
    train_y = train[1]
    test_x = test[0]
    test_y = test[1]
    return train_x, train_y, test_x, test_y
     

if __name__ == '__main__':
    data_file = "mnist.pkl.gz"
    print('Reading training and testing data...')
    train_x, train_y, test_x, test_y = read_data(data_file)
    test_classifiers = ['LR', 'LRM', 'SVM', 'SVMP']
    classifiers = {'LR':logistic_regression_classifier,
                   'LRM':logistic_regression_classifier_multi,
                   'SVM':svm_classifier,
                   'SVMP':svm_poly_classifier
    }
    num_train, num_feat = train_x.shape
    num_test, num_feat = test_x.shape
    print('******************** Data Info *********************')
    print('Training data: %d | Test data: %d | Dimension: %d' % (num_train, num_test, num_feat))
    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        print('Training took %.6fs' % (time.time() - start_time))
        predict = model.predict(test_x)
        accuracy = accuracy_score(test_y, predict)
        print('Test accuracy: %.6f' % (accuracy))
