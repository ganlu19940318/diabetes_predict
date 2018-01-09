# coding=utf-8
import numpy as np
import xgboost as xgb
import pandas as pd
import utils as tl
import matplotlib.pyplot as plt
import operator

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w', encoding="utf-8")
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

# 1. 传统方法
# # data = tl.load_good_data("data/d_train_20180102.csv")
# data = tl.load_all_data_2("data/d_train_20180102.csv")
# # 数据预处理
# data = tl.pre_process(data)
#
# # 构造测试集
# X, y = tl.convert_data_to_featrue_label(data)
#
# # 随机切分数据, 一定要指定随机数种子
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 2. 新方法
# data_set,test_set = tl.split_data("data/d_train_20180102.csv","data/d_train_20180102.csv")
data_set,test_set = tl.split_data("data/d_train_20180102_new.csv","data/d_train_20180102_new.csv")
del data_set["乙肝表面抗原"]
del data_set["乙肝表面抗体"]
del data_set["乙肝e抗原"]
del data_set["乙肝e抗体"]
del data_set["乙肝核心抗体"]
del test_set["乙肝表面抗原"]
del test_set["乙肝表面抗体"]
del test_set["乙肝e抗原"]
del test_set["乙肝e抗体"]
del test_set["乙肝核心抗体"]
X_train, y_train = tl.convert_data_to_featrue_label(data_set)
X_test, y_test = tl.convert_data_to_featrue_label(test_set)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)



# 调参参考
# http://blog.csdn.net/wzmsltw/article/details/50994481

# for i in [0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.3,0.4]:
# for i in [1,2,3,4,5,6,7,8,9]:
param = {
    "seed":4,
    "eval_metric":"rmse",
    "subsample":0.5,
    "booster":"gbtree",
    'max_depth': 2,
    'eta': 0.2,
    'silent': 1,
    'objective': "reg:linear",
    'min_child_weight': 3,
    'gamma': 0
}
# specify validations set to watch performance
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 60
bst = xgb.train(param, dtrain, num_round, watchlist, verbose_eval=False)

# # feature importance
# # 用的时候再打开注释
# features = [x for x in data_set.columns if x not in []]
# ceate_feature_map(features)
# importance = bst.get_fscore(fmap='xgb.fmap')
# importance = sorted(importance.items(), key=operator.itemgetter(1))
# df = pd.DataFrame(importance, columns=['feature', 'fscore'])
# df['fscore'] = df['fscore'] / df['fscore'].sum()
# df.to_csv("temp/feat_importance.csv", index=False,encoding="utf-8")
# df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
# plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt.title('XGBoost Feature Importance')
# plt.xlabel('relative importance')
# plt.show()

# this is prediction
preds = bst.predict(dtest)
labels = dtest.get_label()
print(tl.loss_function(preds,labels))

# 读取比赛题
# exam_set = tl.load_match_data("data/d_test_A_20180102.csv")
exam_set = tl.load_match_data("data/d_test_A_20180102_new.csv")
# 数据预处理
exam_set = tl.pre_process(exam_set)
del exam_set["乙肝表面抗原"]
del exam_set["乙肝表面抗体"]
del exam_set["乙肝e抗原"]
del exam_set["乙肝e抗体"]
del exam_set["乙肝核心抗体"]

# 结果预测
exam_set = xgb.DMatrix(exam_set)
y_exam = bst.predict(exam_set)

# 将结果写入
tl.write_result(y_exam)