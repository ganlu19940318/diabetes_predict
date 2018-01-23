import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import loaddata as ld
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
import time
from sklearn.model_selection import train_test_split
from scipy.stats import skew

# ld.loadgoodData("d_train_20180102")
train = pd.read_csv("d_train_20180102_solve.csv", encoding="gbk", header=0)

# ld.loadgoodData("d_test_A_20180102")
test = pd.read_csv("d_test_A_20180102_solve.csv", header=0, encoding="gbk")

train = train.drop(train[(train['label'] > 30)].index)
train_feat = train.drop('label', axis=1)
train_feat = train_feat.drop('id', axis=1)
test_feat = test
test_feat = test_feat.drop('id', axis=1)


#数据标准化处理
# numeric_feats = train_feat.iloc[:, 0:32].columns
# skewed_feats = train_feat[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# print("\nSkew in numerical features: \n")
# skewness = pd.DataFrame({'Skew' :skewed_feats})
# skewness = skewness[(abs(skewness.Skew) > 0.75)]
# print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
# from scipy.special import boxcox1p
# skewed_features = skewness.index
# lam = 0.15
# for feat in skewed_features:
#     train_feat[feat] = boxcox1p(train_feat[feat], lam)
#     test_feat[feat] = boxcox1p(test_feat[feat], lam)
# train = pd.get_dummies(train_feat)
# test = pd.get_dummies(test_feat)
def prob_class(preds):
    prediction = np.zeros(preds.shape[0])
    for j in range(preds.shape[0]):
        a = np.max(preds[j, :])
        b = np.where(preds[j, :] == np.max(preds[j, :]))
        prediction[j] = b[0][0]
    return prediction

predictors = [f for f in test_feat.columns if f not in ['label_label']]
# X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.1, random_state=1, stratify=train_y) ## 这里保证分割后y的比例分布与原数据一致


print('开始训练...')
params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_error',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': 5,
    'num_class': 4,
}

# lgb_train = lgb.Dataset(X_train, y_train)
# lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

print('开始CV 5折训练...')
scores = []
t0 = time.time()
train_preds = np.zeros(train_feat.shape[0])
test_preds = np.zeros((test_feat.shape[0], 5))
kf = KFold(len(train_feat), n_folds=5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]
    train_feat2 = train_feat.iloc[test_index]
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['label_label'], categorical_feature=['sex'])
    lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2['label_label'])
    gbm = lgb.train(params,
                    lgb_train1,
                    num_boost_round=300,
                    valid_sets=lgb_train2,
                    verbose_eval=10,
                    early_stopping_rounds=50)
    feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    preds = gbm.predict(train_feat2[predictors])
    train_preds[test_index] = prob_class(preds)
    test_preds[:, i] = prob_class(gbm.predict(test_feat[predictors]))
print('线下得分：    {}'.format(mean_squared_error(train_feat['label_label'], train_preds) * 0.5))
print('CV训练用时{}秒'.format(time.time() - t0))

# print('Start predicting...')
# preds = gbm.predict(X_test, num_iteration=gbm.best_iteration) # 输出的是概率结果




# prediction = []
# for i in range(preds.shape[0]):
#     a = np.max(preds[i, :])
#     b = np.where(preds[i, :] == np.max(preds[i, :]))
#     prediction.append(b[0][0])
# y_test = np.array(y_test)
# y_test = y_test.tolist()
# print('线下得分：    {}'.format(f1_score(y_test, prediction, average = None)))
# print(prediction)
# preds_test = gbm.predict(test_feat, num_iteration=gbm.best_iteration)
# prediction_test = []
# for i in range(preds_test.shape[0]):
#     a = np.max(preds_test[i, :])
#     b = np.where(preds_test[i, :] == np.max(preds_test[i, :]))
#     prediction_test.append(b[0][0])
# print(prediction_test)
# importance = gbm.feature_importance()
# names = gbm.feature_name()
# with open('feature_importance.txt', 'w+') as file:
#     for index, im in enumerate(importance):
#         string = names[index] + ', ' + str(im) + '\n'
#         file.write(string)
#
# #
# # submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,
# #                   index=False, float_format='%.4f')
# print ('predicting, classification error=%f' % (sum( int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))
