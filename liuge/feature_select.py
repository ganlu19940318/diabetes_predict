import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor,ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectFromModel


def select_from_models(features, y):


    importance = pd.DataFrame(index=features.columns)

    times = 10
    for i in range(0, times):
        models = {'adaBoost': AdaBoostRegressor(loss='square', ),
                  'extraTrees': ExtraTreesRegressor(n_jobs=-1),
                  'randomForest': RandomForestRegressor(n_jobs=-1),
                  'decisionTree': DecisionTreeRegressor(),
                  'gradientBoosting': GradientBoostingRegressor(),
                  }

        for model_name, model in models.items():

            model.fit(features, y)

            if model_name in importance.columns:
                importance[model_name] = importance[model_name] + model.feature_importances_
            else:
                importance[model_name] = model.feature_importances_

                # importance = importance+imp

    importance = importance / times
    importance['mean'] = importance.mean(axis=1)
    importance.sort_values(by=['mean'], axis=0, ascending=False, inplace=True)
    return importance


def load_importance(path = 'data/feature_importance_selectFromModel.csv'):
    return  pd.read_csv(path, index_col=['index'], encoding='gbk')

