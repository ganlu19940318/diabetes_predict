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
X = train.iloc[:, :-1]
Y = train.iloc[:, -1]
X = np.array(X)
Y = np.array(Y)
names = train.columns[:len(train.columns)-1]
#递归特征消除
#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LinearRegression
# lr = LinearRegression()
# rfe = RFE(lr, n_features_to_select=1)
# rfe.fit(X,Y)
# print("Features sorted by their rank:")
# a = sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))
# print(type(a))
# print(a)

np.random.seed(0)

ranks = {}

def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))

lr = LinearRegression(normalize=True)
lr.fit(X, Y)
ranks["Linear reg"] = rank_to_dict(np.abs(lr.coef_), names)


ridge = Ridge(alpha=7)
ridge.fit(X, Y)
ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)

lasso = Lasso(alpha=.05)
lasso.fit(X, Y)
a = np.abs(lasso.coef_)
ranks["Lasso"] = rank_to_dict(a, names)


rlasso = RandomizedLasso(alpha=0.04)
rlasso.fit(X, Y)
ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), names)

rfe = RFE(lr, n_features_to_select=5)
rfe.fit(X,Y)
ranks["RFE"] = rank_to_dict(np.array(list(map(float, rfe.ranking_))), names, order=-1)

rf = RandomForestRegressor()
rf.fit(X,Y)
ranks["RF"] = rank_to_dict(rf.feature_importances_, names)

f, pval  = f_regression(X, Y, center=True)
ranks["Corr."] = rank_to_dict(f, names)

mine = MINE()
mic_scores = []

for i in range(X.shape[1]):
    mine.compute_score(X[:,i], Y)
    m = mine.mic()
    mic_scores.append(m)

ranks["MIC"] = rank_to_dict(mic_scores, names)


r = {}
for name in names:
    r[name] = round(np.mean([ranks[method][name]
                             for method in ranks.keys()]), 2)

methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")

print("\t%s" % "\t".join(methods))
for name in names:
    print("%s\t%s" % (name, "\t".join(map(str, [ranks[method][name] for method in methods]))))