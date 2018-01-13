import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing


def raw_data(train_path = 'data/d_train_20180102.csv', test_path = 'data/d_test_A_20180102.csv'):
    """
    :return: 返回原始的train_data&test_data
    """
    train_data = pd.read_csv(train_path,header=0,  encoding='gbk')
    # train_data.to_csv('data/d_train_20180102_utf-8.csv',encoding='utf-8')
    test_data = pd.read_csv(test_path,header=0, encoding='gbk' )
    return train_data,test_data
    # test_data.to_csv('data/d_test_A_20180102_utf-8.csv',encoding='utf-8')
def generated_data(train_path = 'data/train_preprocessing.csv', test_path = 'data/test_preprocessing.csv'):
    """

        :return: 返回预处理过的数据
        """
    train, test = raw_data(train_path, test_path)
def preprocessed_data(train_path = 'data/d_train_20180102.csv', test_path = 'data/d_test_A_20180102.csv'):
    """
    
    :return: 返回预处理过的数据
    """
    train,test = raw_data(train_path,test_path)
    train_id = train.id.values.copy()
    feature_columns = [f for f in train.columns if f not in ['id', '血糖']]
    test_id = test.id.values.copy()
    data = pd.concat([train, test])

    # data.dropna(axis=1,thresh=drop_thresh*data.shape[0],inplace=True)

    data['性别'] = data['性别'].map({'男': 1, '女': 0})
    data['体检日期'] = pd.to_datetime(data['体检日期']).apply(lambda a: a.dayofyear)

    data.fillna(data.median(axis=0), inplace=True)
    scaler = preprocessing.MaxAbsScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    train_feat = pd.DataFrame(data[data.id.isin(train_id)])
    test_feat = pd.DataFrame(data[data.id.isin(test_id)])

    test_feat.drop(labels=['血糖'], axis=1, inplace=True)

    return train_feat, test_feat

