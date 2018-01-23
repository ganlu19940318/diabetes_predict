# coding=utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb
import loaddata as ld
from sklearn.model_selection import cross_val_score




# 数据预处理
# ld.loadgoodData("d_train_20180102")
# train = pd.read_csv("d_train_20180102_solve.csv",encoding="gbk",header=0)
train = pd.read_csv("train_solve.csv", encoding="gbk", header=0)
# train_before_eat = train[train['is_eat'] == 0]
# train_after_eat = train[train['is_eat'] == 1]

# f=open('list.txt','r')
# for line in f.readlines():
#     a.append(line.split(','))
# f.close
# a = a[0]
# del a[-1]


# train_set.describe()

# 构造测试集
# X = train.iloc[:, :-1]
# y = train.iloc[:, -1]
X = train.drop(['label', 'id'], axis=1)
y = train['label']

# def rmse_cv(model, X_train, y):
#     rmse = -cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=5)/2
#     return rmse


损失计算函数
def lossfunc(predict, real):
    return sum(np.square(predict-real))/(2 * len(real))
# 随机切分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



#XGboost开工
d_test = xgb.DMatrix(X_test)
d_train = xgb.DMatrix(X_train, label=y_train)

params = {
        'objective': 'reg:linear',
        'min_child_weight': 1.1,                             #越小越容易过拟合
        'eta': 0.01,
        'colsample_bytree': 0.7,
        'max_depth': 5,                                      #每颗树的最大深度，树高越深，越容易过拟合
        'subsample': 0.7,                                     #样本随机采样，较低的值使得算法更加保守，防止过拟合，但是太小的值也会造成欠拟合
        'gamma': 0.1,                                            #后剪枝时，用于控制是否后剪枝的参数
        'silent': 1,                                           #设置成1则没有运行信息输出，最好是设置为0
        'seed': 1,                                            #随机数的种子
        'lambda':10,
    }

plst = list(params.items())
# specify validations set to watch performance
watchlist = [(d_train, 'train')]

#result = xgb.cv(plst, d_train, num_boost_round = 100, early_stopping_rounds=20, verbose_eval=50, show_stdv=False)
bst = xgb.train(plst, d_train, num_boost_round=3000, evals=watchlist)

y_exam = bst.predict(d_test)
y_test = np.array(y_test)
y_test.tolist()
print(loss)





# d_test = xgb.DMatrix(X_test_after_eat)
# d_train = xgb.DMatrix( X_train_after_eat, label=y_train_after_eat)
#
#
# params = {
#         'objective': 'reg:linear',
#         'min_child_weight': 1.1,                             #越小越容易过拟合
#         'eta': 0.01,
#         'colsample_bytree': 0.7,
#         'max_depth': 5,                                      #每颗树的最大深度，树高越深，越容易过拟合
#         'subsample': 0.7,                                     #样本随机采样，较低的值使得算法更加保守，防止过拟合，但是太小的值也会造成欠拟合
#         'gamma': 0.1,                                            #后剪枝时，用于控制是否后剪枝的参数
#         'silent': 1,                                           #设置成1则没有运行信息输出，最好是设置为0
#         'seed': 0,                                            #随机数的种子
#         'lambda':10,
#     }
#
# plst = list(params.items())
# # specify validations set to watch performance
# watchlist = [(d_train, 'train')]
#
# #result = xgb.cv(plst, d_train, num_boost_round = 100, early_stopping_rounds=20, verbose_eval=50, show_stdv=False)
# bst = xgb.train(plst, d_train, num_boost_round = 3000, evals = watchlist)
# y_exam = bst.predict(d_test)
# y_test = np.array(y_test_after_eat)
# y_test.tolist()
# print(lossfunc(y_exam,y_test))




# ###########具体方法选择##########
# ####3.1决策树回归####
# from sklearn import tree
# model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
# ####3.2线性回归####
# from sklearn import linear_model
# model_LinearRegression = linear_model.LinearRegression()
# ####3.3SVM回归####
# from sklearn import svm
# model_SVR = svm.SVR()
# ####3.4KNN回归####
# from sklearn import neighbors
# model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
# ####3.5随机森林回归####
# from sklearn import ensemble
# model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树
# ####3.6Adaboost回归####
# from sklearn import ensemble
# model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
# ####3.7GBRT回归####
# from sklearn import ensemble
# model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
# ####3.8Bagging回归####
# from sklearn.ensemble import BaggingRegressor
# model_BaggingRegressor = BaggingRegressor()
# ####3.9ExtraTree极端随机树回归####
# from sklearn.tree import ExtraTreeRegressor
# model_ExtraTreeRegressor = ExtraTreeRegressor()


