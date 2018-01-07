# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_all_data(filename):
    """
    author:ganlu
    date:2018-1-6 20:13:59
    读入的是文件中的所有数据
    :param filename:文件名
    :return:文件中的所有数据,以gbk的编码格式读入
    """
    return pd.read_csv(filename, header=0, encoding="gbk")


def pre_process(data):
    """
    author:ganlu
    date:2018-1-6 20:13:59
    对data进行预处理
    :param data:data数据
    :return:预处理后的数据
    """
    # del data["id"]    # id
    # del data["性别"]    # "sex"
    # del data["年龄"]    # "age"
    # del data["体检日期"]  # "date"
    # del data["*天门冬氨酸氨基转换酶"] # "feature1"
    # del data["*丙氨酸氨基转换酶"] # "feature2"
    # del data["*碱性磷酸酶"]    # "feature3"
    # del data["*r-谷氨酰基转换酶"]    # "feature4"
    # del data["*总蛋白"]  # "feature5"
    # del data["白蛋白"]   # "feature6"
    # del data["*球蛋白"]    # "feature7"
    # del data["白球比例"]  # "feature8"
    # del data["甘油三酯"]  # "feature9"
    # del data["总胆固醇"]  # "feature10"
    # del data["高密度脂蛋白胆固醇"] # "feature11"
    # del data["低密度脂蛋白胆固醇"] # "feature12"
    # del data["尿素"]    # "feature13"
    # del data["肌酐"]    # "feature14"
    # del data["尿酸"]    # "feature15"
    # del data["乙肝表面抗原"]    # "feature16"
    # del data["乙肝表面抗体"]    # "feature17"
    # del data["乙肝e抗原"] # "feature18"
    # del data["乙肝e抗体"] # "feature19"
    # del data["乙肝核心抗体"]    # "feature20"
    # del data["白细胞计数"] # "feature21"
    # del data["红细胞计数"] # "feature22"
    # del data["血红蛋白"]  # "feature23"
    # del data["红细胞压积"] # "feature24"
    # del data["红细胞平均体积"]   # "feature25"
    # del data["红细胞平均血红蛋白量"]    # "feature26"
    # del data["红细胞平均血红蛋白浓度"]   # "feature27"
    # del data["红细胞体积分布宽度"] # "feature28"
    # del data["血小板计数"] # "feature29"
    # del data["血小板平均体积"]   # "feature30"
    # del data["血小板体积分布宽度"] # "feature31"
    # del data["血小板比积"] # "feature32"
    # del data["中性粒细胞%"]    # "feature33"
    # del data["淋巴细胞%"] # "feature34"
    # del data["单核细胞%"] # "feature35"
    # del data["嗜酸细胞%"] # "feature36"
    # del data["嗜碱细胞%"] # "feature37"
    # del data["血糖"]    # "label"

    return data

def load_all_data_2(filename):
    """
    author:ganlu
    date:2018-1-6 20:13:59
    读入的是文件中的所有数据,
    将体检日期,id这两列删除,
    性别男用1表示,女用-1表示
    :param filename:文件名
    :return:读入的是文件中的所有数据,以gbk的编码格式读入
    """
    data_train = pd.read_csv(filename, header=0, encoding="gbk")
    del data_train["体检日期"]
    del data_train["id"]
    data_train["性别"] = data_train["性别"].apply(lambda x: 1 if x == "男" else -1)
    return data_train


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


def plot_label(filename):
    """
    author:ganlu
    date:2018-1-6 20:18:42
    将数据切分成特征和标签
    :param filename:文件名
    :return 给样本分类图
    """
    data = load_all_data(filename)
    plot_label2(data)


def plot_label2(data):
    """
    author:ganlu
    date:2018-1-6 20:18:42
    将数据切分成特征和标签
    :param data:数据
    :return 给样本分类图
    """
    data["package"] = data.count(axis=1)
    cols = list(data)
    cols.insert(-1, cols.pop())
    data = data.ix[:, cols]
    lists = {}
    for value in set(data["package"]):
        lists.update({value: len(data[data["package"] == value])})
    print(lists)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.hist(np.array(data.ix[:, "package"]))
    plt.show()


def sort_for_trainset(data):
    """
    author:ganlu
    date:2018-1-6 20:18:42
    根据数据集的缺失情况给数据集分类
    :param filename:要分类的数据
    :return 分类后的数据集
    """
    def format(x):
        if x <= 22:
            x = 0
        elif x <= 27:
            x = 1
        elif x <= 34:
            x = 2
        elif x <= 39:
            x = 3
        else:
            x = 4
        return x
    data["package"] = data.count(axis=1)
    cols = list(data)
    cols.insert(-1, cols.pop())
    data = data.ix[:, cols]
    data["package"] = data["package"].apply(format)
    return data


def split_data(trainfile,testfile):
    """
    author:ganlu
    date:2018-1-6 20:18:42
    根据trainset和testset划分数据
    :param trainfile:trainset位置
    :param testfile:testset位置
    :return 划分后的数据集
        test_set:交叉验证集
        data_set:训练集
    """
    test_set = pd.DataFrame(data=None)
    data_set = pd.DataFrame(data=None)
    data_train = load_all_data(trainfile)
    data_train = sort_for_trainset(data_train)
    data_test = load_all_data(testfile)
    data_test = sort_for_trainset(data_test)
    ratelist = {}
    for i in set(data_test["package"]):
        ratelist.update({i: len(data_test[data_test["package"] == i]) / len(data_test)})

    for item in ratelist.items():
        data_temp = data_train[data_train["package"] == item[0]]
        test_set_temp = data_temp.sample(frac=0.3 * item[1] * len(data_train) / len(data_temp),random_state=4)
        data_set_temp = data_temp.append(test_set_temp).drop_duplicates(keep=False)
        test_set = test_set.append(test_set_temp)
        data_set = data_set.append(data_set_temp)
    return data_set,test_set