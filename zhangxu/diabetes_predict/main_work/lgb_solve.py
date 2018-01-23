import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import loaddata as ld
from scipy.special import boxcox, inv_boxcox

# 用lightgbm进行求解
# ld.loadgoodData("d_train_20180102")
# train = pd.read_csv("train_solve.csv", encoding="gbk", header=0)
#
# ld.loadgoodData("d_test_A_20180102")
# test = pd.read_csv("test_solve.csv", header=0, encoding="gbk")
train, test = ld.data_preprocessing("train", "test")
#注释掉的部分为对血糖标签进行标准化，效果不明显，注释掉了
# lambda_boxcox = 1
# train['label'] = boxcox(train['label'], lambda_boxcox)
# train_feat = train.drop('label', axis=1)
train_feat = train.drop('id', axis=1)
test_feat = test
test_feat = test_feat.drop('id', axis=1)
#注释掉的部分为选取部分特征
# a = list()
# f = open('list.txt', 'r')
# for line in f.readlines():
#     a.append(line.split(','))
# f.close
# a = a[0]
# del a[-1]
predictors = [f for f in test_feat.columns if f not in ['label']]

# 误差计算公式
def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label, pred) * 0.5
    return ('mse', score, False)


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
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['label'])
    lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2['label'])
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

print('线下得分：    {}'.format(mean_squared_error(train_feat['label'], train_preds) * 0.5))
print('CV训练用时{}秒'.format(time.time() - t0))

submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,
                  index=False, float_format='%.4f')