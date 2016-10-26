import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def runplt():
    plt.figure()
    plt.title(u'Apartment Price Curve')
    plt.xlabel(u'Size')
    plt.ylabel(u'Price')
    plt.axis([0, 150, 0, 600])
    plt.grid(True)
    return plt


plt = runplt()
X = [[30], [40], [60], [80], [100], [120], [140]]
y = [[320], [360], [400], [455], [490], [546], [580]]
plt.plot(X, y, 'k.')
plt.show()


model = LinearRegression()
model.fit(X, y)
model.coef_
print('The Price of Size 160 Apartment: %.2fW' % model.predict([[160]])[0])


y_pred = model.predict(X)
plt.plot(X, y, 'k.')
plt.plot(X, y_pred, 'g-')
plt.show()
