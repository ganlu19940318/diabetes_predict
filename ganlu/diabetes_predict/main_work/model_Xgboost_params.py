# coding=utf-8
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import utils as tl

data = tl.load_good_data("data/d_train_20180102.csv")
# 构造测试集
X, y = tl.convert_data_to_featrue_label(data)

# 随机切分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 调参参考
# http://blog.csdn.net/wzmsltw/article/details/50994481
param = {'max_depth': 22, 'eta': 0.1, 'silent': 0, 'objective': "reg:linear", 'min_child_weight': 3, 'gamma': 14}

# specify validations set to watch performance
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 33
bst = xgb.train(param, dtrain, num_round, watchlist)

# this is prediction
preds = bst.predict(dtest)
labels = dtest.get_label()
print(tl.loss_function(preds,labels))