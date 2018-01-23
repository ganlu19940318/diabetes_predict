import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time
import pickle as pickle

def feature(x,minValue,maxValue):
    if x is None:
        return 0
    elif x < minValue:
        return 1
    elif x > maxValue:
        return 3
    else:
        return 2


def lable(x,value0,value1,value2):
    if x < value0:
        return 0
    elif x < value1:
        return 1
    elif x < value2:
        return 1
    else:
        return 3


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model


data = pd.read_csv("data/d_train_20180102.csv",header=0,encoding="gbk")
# only "甘油三酯","总胆固醇","高密度脂蛋白胆固醇","低密度脂蛋白胆固醇","血糖"
data = data.drop(["id","性别","年龄","体检日期","*天门冬氨酸氨基转换酶","*丙氨酸氨基转换酶","*碱性磷酸酶",
                  "*r-谷氨酰基转换酶","*总蛋白","白蛋白","*球蛋白","白球比例","尿素","肌酐","尿酸","乙肝表面抗原",
                  "乙肝表面抗体","乙肝e抗原","乙肝e抗体","乙肝核心抗体","白细胞计数","红细胞计数",
                  "血红蛋白","红细胞压积","红细胞平均体积","红细胞平均血红蛋白量","红细胞平均血红蛋白浓度",
                  "红细胞体积分布宽度","血小板计数","血小板平均体积","血小板体积分布宽度","血小板比积",
                  "中性粒细胞%","淋巴细胞%","单核细胞%","嗜酸细胞%","嗜碱细胞%"],
                 axis=1)
data[["甘油三酯_lable","总胆固醇_lable","高密度脂蛋白胆固醇_lable","低密度脂蛋白胆固醇_lable"]]\
    =data[["甘油三酯","总胆固醇","高密度脂蛋白胆固醇","低密度脂蛋白胆固醇"]]
data["血糖"] = data.pop("血糖")
data["甘油三酯_lable"] = data["甘油三酯_lable"].apply(lambda x: feature(x,0.56,1.7))
data["总胆固醇_lable"] = data["总胆固醇_lable"].apply(lambda x: feature(x,2.9,5.72))
data["高密度脂蛋白胆固醇_lable"] = data["高密度脂蛋白胆固醇_lable"].apply(lambda x: feature(x,0.94,2))
data["低密度脂蛋白胆固醇_lable"] = data["低密度脂蛋白胆固醇_lable"].apply(lambda x: feature(x,1.89,3.1))
data["血糖_lable"] = data["血糖"].apply(lambda x: lable(x,3.9,6.1,11.1))

data = data[data["甘油三酯"].notnull()]

# 随机切分数据
X_train, X_test, y_train, y_test = train_test_split(data.ix[:,:-2], data.ix[:,-1], test_size=0.3)

# model = naive_bayes_classifier(X_train, y_train)
# y_predict = model.predict(X_test)
# fig = plt.figure()
# ax1 = fig.add_subplot(1, 1, 1)
# ax1.scatter(np.arange(len(y_test)),np.array(y_test),color="red",s=1)
# ax1.scatter(np.arange(len(y_predict)),np.array(y_predict),color="green",s=1)
# plt.show()
# print(data.ix[:,-1])
# print(len(y_test))
# print(sum(abs(y_predict-y_test)))

# delete
test_classifiers = ['NB','KNN','LR','RF','DT',"SVM",'SVMCV','GBDT']
classifiers = {'NB': naive_bayes_classifier,
               'KNN': knn_classifier,
               'LR': logistic_regression_classifier,
               'RF': random_forest_classifier,
               'DT': decision_tree_classifier,
               'SVM': svm_classifier,
               'SVMCV': svm_cross_validation,
               'GBDT': gradient_boosting_classifier
               }

print('reading training and testing data...')
train_x, train_y, test_x, test_y = X_train, y_train, X_test, y_test

model_save_file = None
model_save = {}

for classifier in test_classifiers:
    print('******************* %s ********************' % classifier)
    # start_time = time.time()
    model = classifiers[classifier](train_x, train_y)
    # print('training took %fs!' % (time.time() - start_time))
    predict = model.predict(test_x)
    # if model_save_file != None:
    #     model_save[classifier] = model
    # precision = metrics.precision_score(test_y, predict)
    # recall = metrics.recall_score(test_y, predict)
    # print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
    # accuracy = metrics.accuracy_score(test_y, predict)
    # print('accuracy: %.2f%%' % (100 * accuracy))

# if model_save_file != None:
#     pickle.dump(model_save, open(model_save_file, 'wb'))
