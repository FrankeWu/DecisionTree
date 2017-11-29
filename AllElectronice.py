from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing
import numpy as np

from sklearn.externals.six import StringIO

# Read in the csv file and put features into list of dict and list of class label
allElectronicsData = open(r'./data/AllElectronics.csv', 'rt')
reader = csv.reader(allElectronicsData)
headers = next(reader)

print(headers)

featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row)-1])#将分类结果放入labellist列表中
    rowDict = {}
    for i in range(1, len(row)-1):
        #print("row wqk")
        #print(row[i])
        rowDict[headers[i]] = row[i]
        #print('rowdict wqk')
        #print("rowdict",rowDict)
    featureList.append(rowDict)

print(featureList)

# Vetorize features
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList) .toarray()

#print(vec.get_feature_names())#将list列表转换成为一个vec的向量。方便计算机使用。这里是打印出来
#print("dummyX: " + str(dummyX))


print("labelList: " + str(labelList))

# vectorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
#print(dummyY)
print("dummyY: " + str(dummyY))

# Using decision tree for classification
# clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)#两个参数，前一个是矩阵，后面的那个是结果的那一列
print("clf: " + str(clf))


# Visualize model
with open("allElectronicInformationGainOri.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names = vec.get_feature_names(), out_file=f)


#进行预测
oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX))

newRowXArray = np.array(newRowX).reshape((1,-1))#这句话是将newRowX列表转换成为数组的形式

predictedY = clf.predict(newRowXArray)#predict中的参数只能是数组不能是列表
print("predictedY: " + str(predictedY))

