import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
import time
import sys
from load_data import *
from feature_select import *

k = 2

def predict_score(ground_truth, pred):
    return mean_squared_error(pow(ground_truth,k),pow(pred,k))*0.5


def main():
    FEATURN_NUM = 36
    train_data,  test_dataB = preprocessed_data()
    importance = load_importance()
    FEATURE_COLUMNS = list(importance.index[0:FEATURN_NUM])
    print('选用特征：',FEATURE_COLUMNS)
    X = train_data[FEATURE_COLUMNS].values
    y = train_data['血糖'].values

    X_subB = test_dataB[FEATURE_COLUMNS].values
    clf = xgb.XGBRegressor(seed=12)
    grid = [{
             'booster':['gbtree'],
             'learning_rate':[0.18],
             #'min_child_weight':[],
             'max_depth':[3],
             'gamma':[0.1],
             'subsample':[ 0.8],
             'colsample_bytree':[0.7],
             'reg_alpha':[1.0],
             'reg_lambda':[0.8],
             'scale_pos_weight':[1]
             },
            ]

    gridCV = GridSearchCV(estimator = clf, param_grid= grid,
                          scoring = make_scorer(predict_score,greater_is_better=False),
                          iid = False,n_jobs=-1,cv = 6,verbose=2,)
    gridCV.fit(X,pow(y,1/k))
    print("best params:",gridCV.best_params_,'best score:',gridCV.best_score_)

    sub_pred = pow(gridCV.predict(X_subB),k)
    model_name = 'xgb-B'+time.strftime("%m%d%H%M", time.localtime())
    sub_pred = pd.DataFrame(sub_pred).round(3)
    sub_pred.to_csv('sub/' + model_name + '.csv', header=False, index=False, encoding='utf-8')

if __name__ == "__main__":
    main()