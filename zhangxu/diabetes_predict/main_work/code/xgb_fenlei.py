import pandas as pd
import xgboost as xgb
from sklearn import cross_validation, metrics
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from scipy.stats import skew

# ld.loadgoodData("d_train_20180102")
train = pd.read_csv("d_train_20180102_solve.csv", encoding="gbk", header=0)

# ld.loadgoodData("d_test_A_20180102")
test = pd.read_csv("d_test_A_20180102_solve.csv", header=0, encoding="gbk")

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

train_x = train_feat.drop('label_label', axis=1)
train_y = train_feat['label_label']
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=1, stratify=train_y) ## 这里保证分割后y的比例分布与原数据一致


print('开始训练...')
xgb_params = {
    "eta": 0.1,
    "seed": 1,
    "colsample_bytree": 0.8,
    "silent": 1,
    "objective": "multi:softmax",
    "num_class": 4,
    "max_depth": 6,
    "min_child_weight": 1,
    "eval_metric": "merror",
    "lambda": 5,
}

xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_eval = xgb.DMatrix(X_test)

# train
print('Start training...')
watchlist = [(xgb_train, "train")]
model = xgb.train(xgb_params,
                  xgb_train,
                  num_boost_round=1000,
                  evals=watchlist,
                  verbose_eval=10,       #每隔多少步输出一个结果
                  early_stopping_rounds=200)


pred = model.predict(xgb_eval)
print(np.sum(pred == 0))
print(np.sum(pred == 1))
print(np.sum(pred == 2))
print(np.sum(pred == 3))
y_test = np.array(y_test)
print('predicting, classification error=%f' % (sum(int(pred[i]) != y_test[i] for i in range(len(y_test))) / float(len(y_test))))




