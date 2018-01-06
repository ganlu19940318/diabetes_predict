# coding=utf-8

# 用knn训练模型

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import utils as tl

# 读数据
data = tl.load_good_data("data/d_train_20180102.csv")

# 将数据分成特征和标签
X, y = tl.convert_data_to_featrue_label(data)

# 随机切分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# knn开工
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)

# 输出结果
predict = knn.predict(X_test)

# 查看loss值
print(tl.loss_function(predict, y_test))