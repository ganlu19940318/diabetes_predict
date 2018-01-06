# coding=utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb



#日期变换格式23/11/2012变成20121123
def data(x):
    arr = x.split('/')
    return int(arr[2] + arr[1] + arr[0])

# 数据预处理
train = pd.read_csv("d_train_20180102.csv",encoding="gbk",header=0)

train["性别"] = train["性别"].apply(lambda x: 1 if x == "男" else -1)
train["体检日期"] = train["体检日期"].apply(lambda x: data(x))

train.rename(columns={
    "id":"id",
    "性别":"sex",
    "年龄":"age",
    "体检日期":"date",
    "*天门冬氨酸氨基转换酶":"feature1",
    "*丙氨酸氨基转换酶":"feature2",
    "*碱性磷酸酶":"feature3",
    "*r-谷氨酰基转换酶":"feature4",
    "*总蛋白":"feature5",
    "白蛋白":"feature6",
    "*球蛋白":"feature7",
    "白球比例":"feature8",
    "甘油三酯":"feature9",
    "总胆固醇":"feature10",
    "高密度脂蛋白胆固醇":"feature11",
    "低密度脂蛋白胆固醇":"feature12",
    "尿素":"feature13",
    "肌酐":"feature14",
    "尿酸":"feature15",
    "乙肝表面抗原":"feature16",
    "乙肝表面抗体":"feature17",
    "乙肝e抗原":"feature18",
    "乙肝e抗体":"feature19",
    "乙肝核心抗体":"feature20",
    "白细胞计数":"feature21",
    "红细胞计数":"feature22",
    "血红蛋白":"feature23",
    "红细胞压积":"feature24",
    "红细胞平均体积":"feature25",
    "红细胞平均血红蛋白量":"feature26",
    "红细胞平均血红蛋白浓度":"feature27",
    "红细胞体积分布宽度":"feature28",
    "血小板计数":"feature29",
    "血小板平均体积":"feature30",
    "血小板体积分布宽度":"feature31",
    "血小板比积":"feature32",
    "中性粒细胞%":"feature33",
    "淋巴细胞%":"feature34",
    "单核细胞%":"feature35",
    "嗜酸细胞%":"feature36",
    "嗜碱细胞%":"feature37",
    "血糖":"label"
}, inplace =True)
#选取特征
a = ['age', 'feature9', 'sex', 'feature6', 'feature7', 'feature21', 'feature5',\
     'feature8', 'feature11', 'feature10', 'feature22', 'feature19', 'feature24', 'feature28', 'feature26', 'feature23', 'feature1']
# f=open('list.txt','r')
# for line in f.readlines():
#     a.append(line.split(','))
# f.close
# a = a[0]
# del a[-1]


# train_set.describe()

# 构造测试集
X = train.iloc[:, :-1]
y = train.iloc[:, -1]



# 损失计算函数
def lossfunc(predict, real):
    return sum(np.square(predict-real))/(2 * len(real))

# 随机切分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train = X_train[a]
X_test = X_test[a]


# print(lossfunc(knn.predict(X_test),y_test))

#XGboost开工
d_test = xgb.DMatrix(X_test)
d_train = xgb.DMatrix( X_train, label=y_train)

params = {
        'objective': 'reg:linear',
        'min_child_weight': 1.1,                             #越小越容易过拟合
        'eta': 0.01,
        'colsample_bytree': 0.7,
        'max_depth': 5,                                      #每颗树的最大深度，树高越深，越容易过拟合
        'subsample': 0.7,                                     #样本随机采样，较低的值使得算法更加保守，防止过拟合，但是太小的值也会造成欠拟合
        'gamma': 0.1,                                            #后剪枝时，用于控制是否后剪枝的参数
        'silent': 1,                                           #设置成1则没有运行信息输出，最好是设置为0
        'seed': 0,                                            #随机数的种子
        'lambda':10,
    }

plst = list(params.items())
# specify validations set to watch performance
watchlist = [(d_train, 'train')]

#result = xgb.cv(plst, d_train, num_boost_round = 100, early_stopping_rounds=20, verbose_eval=50, show_stdv=False)
bst = xgb.train(plst, d_train, num_boost_round = 3000, evals = watchlist)
y_exam = bst.predict(d_test)
y_test = np.array(y_test)
y_test.tolist()
print(lossfunc(y_exam,y_test))




#读取比赛题
# exam_set = pd.read_csv("d_test_A_20180102.csv",header=0,encoding="gbk")
# exam_set["性别"]=exam_set["性别"].apply(lambda x:1 if x == "男" else -1)
# exam_set["体检日期"]=exam_set["体检日期"].apply(lambda x: date(x))
# exam_set.rename(columns={
#     "id":"id",
#     "性别":"sex",
#     "年龄":"age",
#     "体检日期":"date",
#     "*天门冬氨酸氨基转换酶":"feature1",
#     "*丙氨酸氨基转换酶":"feature2",
#     "*碱性磷酸酶":"feature3",
#     "*r-谷氨酰基转换酶":"feature4",
#     "*总蛋白":"feature5",
#     "白蛋白":"feature6",
#     "*球蛋白":"feature7",
#     "白球比例":"feature8",
#     "甘油三酯":"feature9",
#     "总胆固醇":"feature10",
#     "高密度脂蛋白胆固醇":"feature11",
#     "低密度脂蛋白胆固醇":"feature12",
#     "尿素":"feature13",
#     "肌酐":"feature14",
#     "尿酸":"feature15",
#     "乙肝表面抗原":"feature16",
#     "乙肝表面抗体":"feature17",
#     "乙肝e抗原":"feature18",
#     "乙肝e抗体":"feature19",
#     "乙肝核心抗体":"feature20",
#     "白细胞计数":"feature21",
#     "红细胞计数":"feature22",
#     "血红蛋白":"feature23",
#     "红细胞压积":"feature24",
#     "红细胞平均体积":"feature25",
#     "红细胞平均血红蛋白量":"feature26",
#     "红细胞平均血红蛋白浓度":"feature27",
#     "红细胞体积分布宽度":"feature28",
#     "血小板计数":"feature29",
#     "血小板平均体积":"feature30",
#     "血小板体积分布宽度":"feature31",
#     "血小板比积":"feature32",
#     "中性粒细胞%":"feature33",
#     "淋巴细胞%":"feature34",
#     "单核细胞%":"feature35",
#     "嗜酸细胞%":"feature36",
#     "嗜碱细胞%":"feature37",
#     "血糖":"label"
# }, inplace =True)
#
# newexam_set = exam_set[fi.a]
# newexam_set = newexam_set.fillna(newexam_set.mean())
# newexam_set1 = xgb.DMatrix(newexam_set)
# y_exam = pd.DataFrame()
# y_exam['label'] = bst.predict(newexam_set1)
# y_exam.to_csv("submit_result.csv", index=False,encoding="utf-8",header=False)

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


