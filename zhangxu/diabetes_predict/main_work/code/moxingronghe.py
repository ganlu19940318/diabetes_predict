import numpy as np
import loaddata as ld
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
from scipy.stats import norm, skew
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import datetime


# color = sns.color_palette()
# sns.set_style('darkgrid')
# ld.loadgoodData("d_train_20180102")
# train = pd.read_csv("d_train_20180102_solve.csv", encoding="gbk", header=0)
# ld.loadgoodData("d_test_A_20180102")
# test = pd.read_csv("d_test_A_20180102_solve.csv",encoding="gbk", header=0)
# train = train.drop(train[(train['label'] > 30)].index)
# train = train.drop(['feature24', 'feature25', 'feature12', 'feature29'], axis=1)
# test = test.drop(['feature24', 'feature25', 'feature12', 'feature29'], axis=1)





# numeric_feats = train.iloc[:, 0:32].columns
# skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# print("\nSkew in numerical features: \n")
# skewness = pd.DataFrame({'Skew' :skewed_feats})
#
# skewness = skewness[(abs(skewness.Skew) > 0.75)]
# print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
#
# from scipy.special import boxcox1p
# skewed_features = skewness.index
# lam = 0.15
# for feat in skewed_features:
#     train[feat] = boxcox1p(train[feat], lam)
#     test[feat] = boxcox1p(test[feat], lam)
#
#
#
#
# corrmat = train.corr()
# fig = plt.figure(figsize=(14, 14))
# ax1 = fig.add_subplot(111)
# k = 20 #number of variables for heatmap
# cols = corrmat.nlargest(k, 'label')['label'].index
# cm = corrmat[cols].loc[cols]
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# ax1.set_xticklabels(cols.values, rotation='vertical')
# ax1.set_yticklabels(cols.values, rotation='horizontal')
# plt.show()

# from pylab import mpl
# mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
# corrmat = train.corr()
# fig = plt.figure(figsize=(14,14))
# plt.clf()
# ax1 = fig.add_subplot(111)
# sns.heatmap(corrmat, vmax=0.9, square=True)
# ax1.set_xticklabels(corrmat.index, rotation='vertical')
# ax1.set_yticklabels(corrmat.columns, rotation='horizontal')
# plt.show()

# fig = plt.figure()
# fig.set_figheight(6)
# fig.set_figwidth(12)
# ax1 = fig.add_subplot(1, 2, 1)
# ax2 = fig.add_subplot(1, 2, 2)
# ax1.hist(train.label)
# ax2.hist(np.log1p(train.label))
# plt.show()


# fig, ax = plt.subplots()
# ax.scatter(x = train['feature37'], y = train['label'])
# plt.ylabel('label', fontsize=13)
# plt.xlabel('feature1', fontsize=13)
# plt.show()





# train = train[['feature1_label','feature2_label','feature3_label','feature4_label','feature5_label','feature6_label','feature7_label','feature8_label','feature9_label','feature10_label',
#                'feature11_label','feature12_label','feature13_label','feature14_label','feature15_label','feature21_label','feature22_label','feature23_label','feature24_label','feature25_label',
#                'feature26_label','feature27_label','feature28_label','feature29_label','feature30_label','feature31_label','feature32_label','feature33_label','feature34_label','feature35_label',
#                'feature36_label','feature37_label','label_label']]
# dataset = train.apply(lambda x: x.value_counts(), axis = 1).fillna(0)
# support = 0.059
# confidence = 0.75
# find_rule(dataset, support, confidence, ms = '---')

# from scipy.special import boxcox, inv_boxcox
# tmp = boxcox(train['label'], -1.5)
# sns.distplot(tmp, fit=norm)
#
# (mu, sigma) = norm.fit(tmp)
#
# print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#
# #Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
# plt.ylabel('Frequency')
# plt.title('label distribution')
#
# fig = plt.figure()
# res = stats.probplot(tmp, plot=plt)
# plt.show()


# 模型融合

from sklearn.utils import shuffle
# train, test = ld.data_preprocessing("train", "test")
train = pd.read_csv("train_solve.csv", encoding="gbk", header=0)
test = pd.read_csv("test_solve.csv", encoding="gbk", header=0)
test_features = test.drop('id', axis=1)
train_features = train.drop(['label', 'id'], axis=1)
train_labels = train['label']
train_features, train_labels = shuffle(train_features, train_labels, random_state=5)
from sklearn.model_selection import cross_val_score

def rmse_cv(model, X_train, y):
    rmse = -cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=5)/2
    return rmse

model_lgb = lgb.LGBMRegressor(objective='regression',
                              boosting_type='gbdt',
                              n_estimators=10,
                              learning_rate=0.01,
                              num_leaves=60,
                              feature_fraction=0.7,
                              num_boost_round=1000)
model_lgb = model_lgb.fit(train_features, train_labels)
# lgb_rmse = rmse_cv(model_lgb, train_features, train_labels)
# print("{:.5f}:+/-{:.5f}".format(lgb_rmse.mean(), lgb_rmse.std()))

model_xgb = xgb.XGBRegressor(n_estimators=700,
                             max_depth=5,
                             learning_rate=0.02,
                             subsample=0.7,
                             colsample_bytree=0.7,
                             gamma=0.1,
                             reg_lambda=10,
                             seed=1,
                             num_boost_round=1000)
model_xgb = model_xgb.fit(train_features, train_labels)
# xgb_rmse = rmse_cv(xlf, train_features, train_labels)
# print("{:.5f}:+/-{:.5f}".format(xgb_rmse.mean(), xgb_rmse.std()))


from sklearn.ensemble import GradientBoostingRegressor
model_grb = GradientBoostingRegressor(n_estimators=300,
                                      learning_rate=0.1,
                                      max_depth=3,
                                      min_samples_split=20,
                                      min_samples_leaf=5)
model_grb = model_grb.fit(train_features, train_labels)
# grb_rmse = rmse_cv(model_grb, train_features, train_labels)
# print("{:.5f}:+/-{:.5f}".format(grb_rmse.mean(), grb_rmse.std()))

# from sklearn.linear_model import ElasticNet
# model_elastic = ElasticNet()
# model_elastic = model_elastic.fit(train_features, train_labels)
# y = model_elastic.predict(test_features)
# plt.show()
# alphas = np.logspace(-4, -2, 10)
# cv_elastic = [rmse_cv(ElasticNet(alpha=alpha), train_features, train_labels).mean() for alpha in alphas]
# print(alphas, cv_elastic)
# cv_elastic = pd.Series(cv_elastic, index=alphas)
# cv_elastic.plot()
# plt.show()

from sklearn.linear_model import Ridge
model_ridge = Ridge(alpha=30)
model_ridge = model_ridge.fit(train_features, train_labels)
# alphas = [1, 3, 5, 10, 15, 20, 30, 50, 100]
# cv_ridge = [rmse_cv(Ridge(alpha=alpha), train_features, train_labels).mean() for alpha in alphas]
# print(alphas, cv_ridge)
# cv_ridge = pd.Series(cv_ridge, index=alphas)
# cv_ridge.plot()
# plt.show()

from sklearn.linear_model import Lasso
model_lasso = Lasso()
model_lasso = model_lasso.fit(train_features, train_labels)
# alphas = np.logspace(-5, -3, 10)
# cv_lasso = [rmse_cv(Lasso(alpha=alpha), train_features, train_labels).mean() for alpha in alphas]
# print(cv_lasso)
# cv_lasso = pd.Series(cv_lasso, index=alphas)
# cv_lasso.plot()
# plt.show()

# 运行不出结果
# from sklearn.svm import SVR
# svm_model = SVR()
# Cs = [0.008, 0.01, 0.03]
# cv_svm = [rmse_cv(SVR(C=c, kernel='rbf'), train_features, train_labels).mean() for c in Cs]
# print(cv_svm)
# cv_svm = pd.Series(cv_svm, index=Cs)
# cv_svm.plot()
# plt.show()


from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, weights):
        self.models = models
        self.weights = np.array(weights)

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.sum(self.weights * predictions, axis=1)


for model in [model_grb, model_lgb, model_xgb, model_lasso, model_ridge]:
    model.fit(train_features, train_labels)

grb_pred = model_grb.predict(test_features)
lgb_pred = model_lgb.predict(test_features)
xgb_pred = model_xgb.predict(test_features)
lasso_pred = model_lasso.predict(test_features)
ridge_pred = model_ridge.predict(test_features)
prediction = np.array([0.1*grb_pred, 0.5*lgb_pred, 0.1*xgb_pred, 0.15*lasso_pred, 0.15*ridge_pred])
pred = np.sum(prediction, axis=0)
pred_dataframe = pd.DataFrame(pred)
pred_dataframe.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None, index=False, float_format='%.4f')

preds = pd.DataFrame({"grb": grb_pred, "lgb": lgb_pred, "xgb": xgb_pred, "lasso": lasso_pred, "ridge": ridge_pred})
sns.pairplot(preds)
plt.show()

model_aver = AveragingModels(models=(model_grb, model_lgb, model_xgb, model_lasso, model_ridge),
                             weights=(0.1, 0.5, 0.1, 0.15, 0.15))

aver_rmse = rmse_cv(model_aver, train_features, train_labels)

print(aver_rmse.mean())