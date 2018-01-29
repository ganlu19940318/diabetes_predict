import numpy as np
import pandas as pd

from sklearn import preprocessing


def raw_data(train_path = 'data/d_train_20180102.csv',  test_pathB = 'data/d_test_B_20180128.csv' ):
    """
    :return: 返回原始的train_data&test_data
    """
    train_data = pd.read_csv(train_path,header=0,  encoding='gbk')

    test_dataB=  pd.read_csv(test_pathB,header=0, encoding='gbk' )
    return train_data ,test_dataB



def generated_data(train_path = 'data/train_preprocessing.csv', test_pathB = 'data/d_test_B_20180128.csv'):
    """

        :return: 返回预处理过的数据
        """
    train, test = raw_data(train_path,  test_pathB)

def outlier_handler(data):
    return 1

def preprocessed_data(train_path = 'data/d_train_20180102.csv', test_path = 'data/d_test_B_20180128.csv'):
    """
    
    :return: 返回预处理过的数据
    """
    train, testB = raw_data(train_path,test_path)
    train_id = train.id.values.copy()
    feature_columns = [f for f in train.columns if f not in ['id', '血糖']]

    test_idB = testB.id.values.copy()
    data = pd.concat([train,   testB])

    # 类别映射
    data['性别'] = data['性别'].map({'男': 1, '女': 0})
    # 日期映射
    data['体检日期'] = pd.to_datetime(data['体检日期']).apply(lambda a: a.dayofyear)

    # 缺失值处理
    data.fillna(data.median(axis=0), inplace=True)

    # 归一化
    scaler = preprocessing.MaxAbsScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])

    train_feat = pd.DataFrame(data[data.id.isin(train_id)])

    test_featB = pd.DataFrame(data[data.id.isin(test_idB)])
    test_featB.drop(labels=['血糖'], axis=1, inplace=True)

    return train_feat,  test_featB


