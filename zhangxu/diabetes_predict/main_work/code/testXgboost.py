# coding=utf-8

# 1.导入相关库，读取数据
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import loaddata as ld

#记录程序运行时间
# import time
# start_time = time.time()
# print(start_time)

data = ld.loadgoodData()
# 构造测试集
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 随机切分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 损失计算函数
def lossfunc(predict, real):
    return sum(np.square(predict-real))/(2 * len(real))

reg = XGBRegressor()

# 监控数据
# eval_set = [(X_test, y_test)]
# eval_metric监控用到的loss function
# verbose开启监控
# reg.fit(X_train, y_train,early_stopping_rounds=1000000, eval_metric="logloss", eval_set=eval_set, verbose=True)

reg.fit(X_train, y_train)
y_predict = reg.predict(X_test)
print(lossfunc(y_predict,y_test))


