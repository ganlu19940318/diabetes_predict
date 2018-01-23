# coding=utf-8

import pandas as pd
import xgboost as xgb
import operator
import matplotlib.pyplot as plt
import numpy as np
import loaddata as ld
#生成xgb.fmap
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w', encoding="utf-8")
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()
#提取id之前的特征
def new_train(s):
    a = np.array(s)
    a = a.tolist()
    a = a[::-1]
    newtrain = train[a]
    return newtrain, a
ld.loadgoodData("d_train_20180102")
train = pd.read_csv("d_train_20180102_solve.csv",encoding="gbk",header=0)


params = {
    'min_child_weight': 1.1,
    'eta': 0.02,
    'colsample_bytree': 0.7,
    'max_depth': 5,
    'subsample': 0.7,
    'lambda':10,
    'gamma': 0.1,
    'silent': 1,
    'verbose_eval': True,
    'seed': 2
}



rounds = 1000





y = train['label']
X = train.drop(['label'], axis=1)

xgtrain = xgb.DMatrix(X, label=y)
bst = xgb.train(params, xgtrain, num_boost_round=rounds)

features = [x for x in train.columns if x not in []]

ceate_feature_map(features)

importance = bst.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()


newtrain, a = new_train(df['feature'])
newtrain['label'] = train['label']
#把特征生成txt文件
fl=open('list.txt', 'w')
for i in a:
    fl.write(i)
    fl.write(",")
fl.close()
#绘制出条形图
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.show()