from numpy import *
import operator
from sklearn.naive_bayes import GaussianNB
import pickle

def readData(path):
    with open(path) as fp:
        lines = fp.readlines()
        row = len(lines)
        col = len(lines[0].split(',')) - 2
        X = zeros((row, col))
        y = []
        index = 0
        for line in lines:
            line_split = line.split(',')
            X[index,:] = line_split[0 :col]
            y.append(int(line_split[-1]))
            index += 1
    return X, y


def classify(inX, dataSet, labels, k):
    dataLen = len(labels)
    distances = (((tile(inX, (dataLen, 1)) - dataSet)**2).sum(1))**0.5
    classCnt = {}
    disSortedIndex = distances.argsort()
    for i in range(k):
        votelabel = labels[disSortedIndex[i]]
        classCnt[votelabel] = classCnt.get(votelabel, 0) + 1
    sortedClassCnt = sorted(classCnt.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCnt[0][0]


X_train, Y_train = readData('./optdigits.tra')
X_test, Y_test = readData('./optdigits.tes')

clf = GaussianNB()
clf.fit(X_train,Y_train)
predictions = clf.predict(X_train)
print ("navie bayes train error :",1.0*sum(predictions != Y_train)/len(Y_train))
predictions = clf.predict(X_test)
print ("navie bayes test error :",1.0*sum(predictions != Y_test)/len(Y_test))

# save the classifier
with open('nb.pkl', 'wb') as fp:
    pickle.dump(clf, fp)

# load it again
with open('nb.pkl', 'rb') as fp:
    clf = pickle.load(fp)


index = -1
cnt = 0
for i in Y_train:
    index += 1
    if i != classify(X_train[index], X_train, Y_train, 10):
        cnt += 1
print ("knn train error :",1.0*cnt/len(Y_train))


index = -1
cnt = 0
for i in Y_test:
    index += 1
    if i != classify(X_test[index], X_train, Y_train, 10):
        cnt += 1
print ("knn test error :",1.0*cnt/len(Y_test))
