import numpy as np
from sklearn.naive_bayes import BernoulliNB

rng = np.random.RandomState(1)
X = rng.randint(5, size=(6, 100))
Y = np.arrat([1, 2, 3, 4, 5])
clf = BernoulliNB()
clf.fit(X, Y)
# BernoulliNB()
print(clf.predict(X[2:3]))