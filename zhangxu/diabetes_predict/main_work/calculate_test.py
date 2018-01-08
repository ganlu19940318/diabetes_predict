# coding=utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import loaddata as ld


# 数据预处理
ld.loadgoodData("d_train_20180102")
train = pd.read_csv("d_train_20180102_solve.csv",encoding="gbk",header=0)


# 构造测试集
X = train.drop('label', axis = 1)
y = train[['id', 'label','is_eat']]


# def group(df, eat, healthy):
#     a = df[df['is_eat'] == eat]
#     X_train_before_eat_healthy = a[a['label_label'] == healthy]
#     X_train_before_eat_healthy = X_train_before_eat_healthy.drop(['is_eat', 'label_label'], axis = 1)
#     return X_train_before_eat_healthy
#
# X_before_eat_healthy = group(X, 0, 0)
# X_before_eat_unhealthy = group(X, 0, 1)
# X_after_eat_healthy = group(X, 1, 0)
# X_after_eat_unhealthy = group(X, 1, 1)
#
# y_before_eat_healthy = group(y, 0, 0)['label']
# y_before_eat_unhealthy = group(y, 0, 1)['label']
# y_after_eat_healthy = group(y, 1, 0)['label']
# y_after_eat_unhealthy = group(y, 1, 1)['label']

X_before = X[X['is_eat'] == 0]
X_after = X[X['is_eat'] == 1]

y_before = y[y['is_eat'] == 0]['label']
y_after = y[y['is_eat'] == 1]['label']






# print(lossfunc(knn.predict(X_test),y_test))

#XGboost开工
d_train = xgb.DMatrix( X_before, label=y_before)
#xgboost参数
params = {
        'objective': 'reg:linear',
        'min_child_weight': 1.1,                             #越小越容易过拟合
        'eta': 0.01,
        'colsample_bytree': 0.7,
        'max_depth': 5,                                      #每颗树的最大深度，树高越深，越容易过拟合
        'subsample': 0.7,                                     #样本随机采样，较低的值使得算法更加保守，防止过拟合，但是太小的值也会造成欠拟合
        'gamma': 1,                                            #后剪枝时，用于控制是否后剪枝的参数
        'silent': 1,                                           #设置成1则没有运行信息输出，最好是设置为0
        'seed': 1,                                            #随机数的种子
        'lambda':10,
    }

plst = list(params.items())
watchlist = [(d_train, 'train')]
#交叉验证
# result = xgb.cv(plst, d_train, num_boost_round = 1000, early_stopping_rounds=200, verbose_eval=50)
#训练模型
bst1 = xgb.train(plst, d_train, num_boost_round = 3000, evals = watchlist)


#读取比赛题
ld.loadgoodData("d_test_A_20180102")
exam_set = pd.read_csv("d_test_A_20180102_solve.csv",header=0,encoding="gbk")


# exam_set_before_eat_healthy = group(exam_set, 0, 0)
# exam_set_before_eat_unhealthy = group(exam_set, 0, 1)
# exam_set_after_eat_healthy = group(exam_set, 1, 0)
# exam_set_after_eat_unhealthy = group(exam_set, 1, 1)

exam_set_before = exam_set[exam_set['is_eat'] == 0]

exam_set_after = exam_set[exam_set['is_eat'] == 1]
newexam_set = xgb.DMatrix(exam_set_before)

exam_set_before['label'] = bst1.predict(newexam_set)
y_a = exam_set_before[['id', 'label']]



d_train = xgb.DMatrix( X_after, label=y_after)
#xgboost参数
params = {
        'objective': 'reg:linear',
        'min_child_weight': 1.1,                             #越小越容易过拟合
        'eta': 0.01,
        'colsample_bytree': 0.7,
        'max_depth': 5,                                      #每颗树的最大深度，树高越深，越容易过拟合
        'subsample': 0.7,                                     #样本随机采样，较低的值使得算法更加保守，防止过拟合，但是太小的值也会造成欠拟合
        'gamma': 1,                                            #后剪枝时，用于控制是否后剪枝的参数
        'silent': 1,                                           #设置成1则没有运行信息输出，最好是设置为0
        'seed': 0,                                            #随机数的种子
        'lambda':10,
    }

plst = list(params.items())
watchlist = [(d_train, 'train')]
#交叉验证
# result = xgb.cv(plst, d_train, num_boost_round = 1000, early_stopping_rounds=200, verbose_eval=50)
#训练模型
bst2 = xgb.train(plst, d_train, num_boost_round = 3000, evals = watchlist)
newexam_set = xgb.DMatrix(exam_set_after)
exam_set_after['label'] = bst2.predict(newexam_set)
y_b = exam_set_after[['id', 'label']]
y_exam = pd.concat([y_a, y_b], axis = 0)
y_exam = y_exam.sort_values(by = 'id')
y_exam = y_exam['label']




y_exam.to_csv("submit_result.csv", index=False,encoding="utf-8",header=False)