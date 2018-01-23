# coding: utf-8
# pylint: disable = invalid-name, C0111
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
import loaddata as ld

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError('You need to install matplotlib for plot_example.py.')

# load or create your dataset
ld.loadgoodData("d_train_20180102")
train = pd.read_csv("d_train_20180102_solve.csv",encoding="gbk",header=0)
X = train.drop('label', axis = 1)
y = train['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=2)



# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
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
}

evals_result = {}  # to record eval results for plotting

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=[lgb_train, lgb_test],
                categorical_feature=['sex'],
                verbose_eval=100)



print('Plot feature importances...')
ax = lgb.plot_importance(gbm, max_num_features=50)
plt.show()