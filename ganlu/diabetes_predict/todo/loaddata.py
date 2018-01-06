# coding=utf-8

import pandas as pd

def loadAllData():
    return pd.read_csv("d_train_20180102.csv",header=0,encoding="gbk")

def loadgoodData():
    data_train = pd.read_csv("d_train_20180102.csv", header=0, encoding="gbk")
    del data_train["体检日期"]
    del data_train["id"]
    data_train["性别"] = data_train["性别"].apply(lambda x: 1 if x == "男" else -1)
    for col in data_train.columns:
        data_train = data_train[data_train[col].notnull()]
    return data_train

