# coding=utf-8

import pandas as pd
import xgboost as xgb
import operator
import matplotlib.pyplot as plt
import numpy as np

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w', encoding="utf-8")
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def new_train(s):
    a = np.array(s)
    a = a.tolist()
    a = a[::-1]
    n = a.index('id')
    a = a[:n]
    newtrain = train[a]
    return newtrain, a

train = pd.read_csv("d_train_20180102.csv",encoding="gbk",header=0)

params = {
    'min_child_weight': 1.1,
    'eta': 0.01,
    'colsample_bytree': 0.7,
    'max_depth': 5,
    'subsample': 0.7,
    'lambda':10,
    'gamma': 0.1,
    'silent': 1,
    'verbose_eval': True,
    'seed': 12
}

def data(x):
    arr = x.split('/')
    return int(arr[2] + arr[1] + arr[0])

rounds = 1000

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
#df.to_csv("feat_importance.csv", index=False,encoding="utf-8")

newtrain, a = new_train(df['feature'])
newtrain['label'] = train['label']

fl=open('list.txt', 'w')
for i in a:
    fl.write(i)
    fl.write(",")
fl.close()

#df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
# plt.title('XGBoost Feature Importance')
# plt.xlabel('relative importance')
# plt.show()