# coding=utf-8
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import utils as tl
import matplotlib.pyplot as plt

# data = tl.load_good_data("data/d_train_20180102.csv")
data = tl.load_all_data_2("data/d_train_20180102.csv")
# 数据预处理
data = tl.pre_process(data)

# 构造测试集
X, y = tl.convert_data_to_featrue_label(data)

# 随机切分数据, 一定要指定随机数种子
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 调参参考
# http://blog.csdn.net/wzmsltw/article/details/50994481

param = {"seed":4,"eval_metric":"rmse","subsample":0.5,"booster":"gbtree",'max_depth': 20, 'eta': 0.1, 'silent': 0, 'objective': "reg:linear", 'min_child_weight': 9, 'gamma': 25}

# specify validations set to watch performance
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 60
bst = xgb.train(param, dtrain, num_round, watchlist, verbose_eval=True)

# this is prediction
preds = bst.predict(dtest)
labels = dtest.get_label()
print(tl.loss_function(preds,labels))

# 读取比赛题
exam_set = tl.load_match_data("data/d_test_A_20180102.csv")
# 数据预处理
exam_set = tl.pre_process(exam_set)

# 结果预测
exam_set = xgb.DMatrix(exam_set)
y_exam = bst.predict(exam_set)

# 将结果写入
tl.write_result(y_exam)