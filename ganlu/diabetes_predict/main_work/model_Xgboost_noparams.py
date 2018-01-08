# coding=utf-8

# 用xgboost训练模型

import utils as tl
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# 读训练集数据
train_set = tl.load_good_data("data/d_train_20180102.csv")

# 将数据分成特征和标签
X, y = tl.convert_data_to_featrue_label(train_set)

# 随机切分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# XGboost开工
reg = XGBRegressor()
reg.fit(X_train, y_train)
y_predict = reg.predict(X_test)
print(tl.loss_function(y_predict, y_test))

# 读取比赛题
exam_set = tl.load_match_data("data/d_test_A_20180102.csv")

# 结果预测
y_exam = reg.predict(exam_set)

# 将结果写入
tl.write_result(y_exam)




