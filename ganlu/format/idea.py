# _*_ coding:utf-8 _*_
import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# 误差计算公式
def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label, pred) * 0.5
    return ('mse', score, False)

# 1. 把train_set分为血糖前100的数据和其他数据,数据分别为train_set1和train_set2
train = pd.read_csv('ganlu_d_train_20180102.csv',header=0,encoding='gbk')
lc = pd.DataFrame(train)
train_set1 = lc.sort_values(["血糖"],ascending=False).head(100)
lc = lc.append(train_set1)
train_set2 = lc.drop_duplicates(keep=False)

# 2. 用train_set训练模型,得到1个模型A1
# ----------------
X_train, X_test, y_train, y_test = train_test_split(train.drop("血糖",axis=1), train["血糖"], test_size=0.3, random_state=1)
train_feat = pd.concat([X_train,y_train],axis=1)
test_feat = pd.concat([X_test,y_test],axis=1)
predictors = [f for f in test_feat.columns if f not in ['血糖']]


print('开始训练...')
params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
    'lambda_l1': 0.4,
    'lambda_l2': 0.5,
}

print('开始CV 5折训练...')
scores = []
t0 = time.time()
train_preds = np.zeros(train_feat.shape[0])
test_preds = np.zeros((test_feat.shape[0], 5))
# 5折交叉验证
kf = KFold(len(train_feat), n_folds=5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]
    train_feat2 = train_feat.iloc[test_index]
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['血糖'])
    lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2['血糖'])
    gbm = lgb.train(params,
                    lgb_train1,
                    num_boost_round=3000,
                    valid_sets=lgb_train2,
                    verbose_eval=100,
                    feval=evalerror,
                    early_stopping_rounds=100)
    feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    train_preds[test_index] += gbm.predict(train_feat2[predictors])
    test_preds[:, i] = gbm.predict(test_feat[predictors])

print('线下得分：    {}'.format(mean_squared_error(train_feat['血糖'], train_preds) * 0.5))
print('CV训练用时{}秒'.format(time.time() - t0))

submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
joblib.dump(gbm,"A1")

# # 3. 用train_set1和train_set2分别训练模型,可以得到2个模型B1,B2
train = train_set1
X_train, X_test, y_train, y_test = train_test_split(train.drop("血糖",axis=1), train["血糖"], test_size=0.3, random_state=1)
train_feat = pd.concat([X_train,y_train],axis=1)
test_feat = pd.concat([X_test,y_test],axis=1)
predictors = [f for f in test_feat.columns if f not in ['血糖']]


print('开始训练...')
params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 1,
    'min_hessian': 1,
    'verbose': -1,
    'lambda_l1': 0.4,
    'lambda_l2': 0.5,
}

print('开始CV 5折训练...')
scores = []
t0 = time.time()
train_preds = np.zeros(train_feat.shape[0])
test_preds = np.zeros((test_feat.shape[0], 5))
# 5折交叉验证
kf = KFold(len(train_feat), n_folds=5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]
    train_feat2 = train_feat.iloc[test_index]
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['血糖'])
    lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2['血糖'])
    gbm = lgb.train(params,
                    lgb_train1,
                    num_boost_round=3000,
                    valid_sets=lgb_train2,
                    verbose_eval=100,
                    feval=evalerror,
                    early_stopping_rounds=100)
    feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    train_preds[test_index] += gbm.predict(train_feat2[predictors])
    test_preds[:, i] = gbm.predict(test_feat[predictors])

print('线下得分：    {}'.format(mean_squared_error(train_feat['血糖'], train_preds) * 0.5))
print('CV训练用时{}秒'.format(time.time() - t0))

submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
joblib.dump(gbm,"B1")

####################

train = train_set2
X_train, X_test, y_train, y_test = train_test_split(train.drop("血糖",axis=1), train["血糖"], test_size=0.3, random_state=1)
train_feat = pd.concat([X_train,y_train],axis=1)
test_feat = pd.concat([X_test,y_test],axis=1)
predictors = [f for f in test_feat.columns if f not in ['血糖']]


print('开始训练...')
params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
    'lambda_l1': 0.4,
    'lambda_l2': 0.5,
}

print('开始CV 5折训练...')
scores = []
t0 = time.time()
train_preds = np.zeros(train_feat.shape[0])
test_preds = np.zeros((test_feat.shape[0], 5))
# 5折交叉验证
kf = KFold(len(train_feat), n_folds=5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]
    train_feat2 = train_feat.iloc[test_index]
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['血糖'])
    lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2['血糖'])
    gbm = lgb.train(params,
                    lgb_train1,
                    num_boost_round=3000,
                    valid_sets=lgb_train2,
                    verbose_eval=100,
                    feval=evalerror,
                    early_stopping_rounds=100)
    feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    train_preds[test_index] += gbm.predict(train_feat2[predictors])
    test_preds[:, i] = gbm.predict(test_feat[predictors])

print('线下得分：    {}'.format(mean_squared_error(train_feat['血糖'], train_preds) * 0.5))
print('CV训练用时{}秒'.format(time.time() - t0))

submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
joblib.dump(gbm,"B2")
# -------------------------------
# 保存结果,暂时不开
# submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,
#                   index=False, float_format='%.4f')

# 4. 用模型A1跑test_set,并将结果排序,取排序结果前1.5%为test_set1,其他为test_set2
# print("hello")

bst = joblib.load("A1")
test_set = pd.read_csv('ganlu_d_train_20180102.csv',header=0,encoding='gbk')
test_set = test_set.drop("血糖",axis=1)
test_set["血糖"] = bst.predict(test_set)
lc = pd.DataFrame(test_set)
test_set1 = lc.sort_values(["血糖"],ascending=False).head(int(len(lc)*0.015))
lc = lc.append(test_set1)
test_set2 = lc.drop_duplicates(keep=False)
test_set1 = test_set1.drop("血糖",axis=1)
test_set2 = test_set2.drop("血糖",axis=1)

# 5. 用B1跑test_set1,用B2跑test_set2
bst = joblib.load("B1")
test_set1["血糖"] = bst.predict(test_set1)
bst = joblib.load("B2")
test_set2["血糖"] = bst.predict(test_set2)

# 6. 将B1,B2的combine
test_set = pd.concat([test_set1,test_set2],axis=0)
print(test_set)




