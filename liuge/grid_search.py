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

def predict_score(ground_truth, pred):
    return mean_squared_error(ground_truth,pred)*0.5


def main():
    FEATURN_NUM = 36
    train_data, test_data = preprocessed_data()
    importance = load_importance()
    FEATURE_COLUMNS = list(importance.index[0:FEATURN_NUM])
    print('选用特征：',FEATURE_COLUMNS)
    X = train_data[FEATURE_COLUMNS].values
    y = train_data['血糖'].values
    X_sub = test_data[FEATURE_COLUMNS].values

    clf = xgb.XGBRegressor()
    grid = [{
             'booster':['gbtree'],
             'learning_rate':[0.01,0.03,0.06,0.1,0.15,0.2],
             # 'min_child_weight':[],
             'max_depth':[3,5,7,9],
             'gamma':[0.01,0.03,0.1,0.3],
             'subsample':[0.5,0.7,0.9],
             'colsample_bytree':[0.5,0.7,0.9],
             'reg_alpha':[0.1,0.3,0.9],
             'reg_lambda':[0.1,0.3,0.9],
             'scale_pos_weight':[0.1,0.3,0.9]
             },
            ]
    gridCV = GridSearchCV(estimator = clf, param_grid= grid,
                          scoring = make_scorer(predict_score,greater_is_better=False),
                          iid = False,n_jobs=-1,cv = 6,verbose=2)
    gridCV.fit(X,y)
    print('scores:',gridCV.grid_scores_,"best params:",gridCV.best_params_,'best score:',gridCV.best_score_)

    sub_pred = gridCV.predict(X_sub)
    model_name = 'xgb-'+time.strftime("%m%d%H%M", time.localtime())
    sub_pred = pd.DataFrame(sub_pred).round(3)
    sub_pred.to_csv('sub/' + model_name + '.csv', header=False, index=False, encoding='utf-8')

if __name__ == "__main__":
    main()