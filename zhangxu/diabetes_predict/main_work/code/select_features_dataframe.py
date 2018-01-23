from sklearn.linear_model import (LinearRegression, Ridge,  Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from minepy import MINE
import loaddata as ld
import pandas as pd

train = ld.loadgoodData()
#pearson系数选择特征
# a = np.round(train.corr(method = 'pearson'), 2)
# a_label = a['label']
# a_label = a_label.sort_values(ascending=False)
# a_label = a_label.index.tolist()
# n = a_label.index('id')
# a_label = a_label[1:n]

#数据的处理
X = train.iloc[:, :-1]
Y = train.iloc[:, -1]
X = np.array(X)
Y = np.array(Y)
names = train.columns[:len(train.columns)-1]
select_feature = pd.DataFrame(index = names)
np.random.seed(0)

def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = list(map(lambda x: round(x, 2), ranks))
    return ranks
#线性回归
lr = LinearRegression(normalize=True)
lr.fit(X, Y)
select_feature["Linear reg"] = rank_to_dict(np.abs(lr.coef_), names)

#岭回归，L2正则化
ridge = Ridge(alpha=7)
ridge.fit(X, Y)
select_feature["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)
#Lasso回归，L1正则化
lasso = Lasso(alpha=.05)
lasso.fit(X, Y)
a = np.abs(lasso.coef_)
select_feature["Lasso"] = rank_to_dict(a, names)

#稳定性选择
rlasso = RandomizedLasso(alpha=0.04)
rlasso.fit(X, Y)
select_feature["Stability"] = rank_to_dict(np.abs(rlasso.scores_), names)
#递归特征消除
rfe = RFE(lr, n_features_to_select=5)
rfe.fit(X,Y)
select_feature["RFE"] = rank_to_dict(np.array(list(map(float, rfe.ranking_))), names, order=-1)
#随机森林
rf = RandomForestRegressor()
rf.fit(X,Y)
select_feature["RF"] = rank_to_dict(rf.feature_importances_, names)
#互信息和最大信息系数
f, pval  = f_regression(X, Y, center=True)
select_feature["Corr."] = rank_to_dict(f, names)

mine = MINE()
mic_scores = []

for i in range(X.shape[1]):
    mine.compute_score(X[:,i], Y)
    m = mine.mic()
    mic_scores.append(m)

select_feature["MIC"] = rank_to_dict(mic_scores, names)

select_feature['mean'] = select_feature.apply(lambda x: x.mean(), axis = 1)
print(select_feature)
a = select_feature.sort_values(by="mean", ascending=False)
a.to_csv("a.csv", index=True,encoding="utf-8",header=True)