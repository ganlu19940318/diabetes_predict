# coding=utf-8

import pandas as pd
import numpy as np


def load_all_data(filename):
    """
    author:ganlu
    date:2018-1-6 20:13:59
    读入的是文件中的所有数据
    :param filename:文件名
    :return:文件中的所有数据,以gbk的编码格式读入
    """
    return pd.read_csv(filename, header=0, encoding="gbk")


def load_match_data(filename):
    """
    author:ganlu
    date:2018-1-6 20:13:59
    读入的是比赛文件中的所有数据
    将体检日期,id这两列删除,
    性别男用1表示,女用-1表示
    :param filename:文件名
    :return:以gbk的编码格式读入
    """
    exam_set = pd.read_csv(filename, header=0, encoding="gbk")
    exam_set["性别"] = exam_set["性别"].apply(lambda x: 1 if x == "男" else -1)
    exam_set["体检日期"] = exam_set["体检日期"].apply(lambda x: date_to_int(x))
    del exam_set["体检日期"]
    del exam_set["id"]
    return exam_set


def load_good_data(filename):
    """
    author:ganlu
    date:2018-1-6 20:13:59
    读入的是文件中的所有无缺损数据,
    将体检日期,id这两列删除,
    性别男用1表示,女用-1表示
    :param filename:文件名
    :return:读入的是文件中的所有无缺损数据,以gbk的编码格式读入
    """
    data_train = pd.read_csv(filename, header=0, encoding="gbk")
    del data_train["体检日期"]
    del data_train["id"]
    data_train["性别"] = data_train["性别"].apply(lambda x: 1 if x == "男" else -1)
    return get_notnull_data(data_train)


def get_notnull_data(data):
    """
    author:ganlu
    date:2018-1-6 20:13:59
    将数据进行清洗
    将数据中无损的数据取出来
    :param data:
    :return:
    """
    for col in data.columns:
        data = data[data[col].notnull()]
    return data


def date_to_int(date):
    """
    author:ganlu
    date:2018-1-6 20:13:59
    :param date:日期,字符串,格式为18/12/2017
    :return:日期的整型值,格式为20171218
    """
    arr = date.split('/')
    return int(arr[2] + arr[1] + arr[0])


def loss_function(predict, real):
    """
    author:ganlu
    date:2018-1-6 20:13:59
    损失计算函数
    :param predict:预测值
    :param real:真实值
    :return:loss值
    """
    return sum(np.square(predict - real)) / (2 * len(real))


def write_result(y_exam):
    """
    author:ganlu
    date:2018-1-6 20:18:42
    将结果写入文件
    :param y_exam:预测结果
    :return
    """
    y_exam = pd.DataFrame(y_exam)
    y_exam = y_exam.round(3)
    y_exam.to_csv("temp/submit_result.csv", index=False, encoding="utf-8", header=False)
    return


def convert_data_to_featrue_label(train_set):
    """
    author:ganlu
    date:2018-1-6 20:18:42
    将数据切分成特征和标签
    :param train_set:数据
    :return X:特征
    :return y:标签
    """
    X = train_set.iloc[:, :-1]
    y = train_set.iloc[:, -1]
    return X, y
