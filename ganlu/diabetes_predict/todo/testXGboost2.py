# coding=utf-8
import numpy as np
import xgboost as xgb
import loaddata as ld
from sklearn.model_selection import train_test_split
import pandas as pd

#没跑通

data = ld.loadgoodData()
# 构造测试集
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 随机切分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

dtrain = pd.concat([X_train, y_train], axis=1)
dtest = pd.concat([X_test, y_test], axis=1)
# specify parameters via map, definition are same as c++ version
param = {'max_depth': 22, 'eta': 0.1, 'silent': 0, 'objective': "reg:linear", 'min_child_weight': 3, 'gamma': 14}

# specify validations set to watch performance
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 33
bst = xgb.train(param, dtrain, num_round, watchlist)

# this is prediction
preds = bst.predict(dtest)
labels = dtest.get_label()
# print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))
#
# print('correct=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) == labels[i]) / float(len(preds))))