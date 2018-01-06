# coding=utf-8

import pandas as pd

def loadAllData():
    return pd.read_csv("d_train_20180102.csv",header=0,encoding="gbk")

def data(x):
    arr = x.split('/')
    return int(arr[2] + arr[1] + arr[0])

def loadgoodData():
    data_train = pd.read_csv("d_train_20180102.csv", header=0, encoding="gbk")
    data_train["性别"] = data_train["性别"].apply(lambda x: 1 if x == "男" else -1)
    data_train["体检日期"] = data_train["体检日期"].apply(lambda x: data(x))
    data_train.rename(columns={
        "id":"id",
        "性别":"sex",
        "年龄":"age",
        "体检日期":"date",
        "*天门冬氨酸氨基转换酶":"feature1",
        "*丙氨酸氨基转换酶":"feature2",
        "*碱性磷酸酶":"feature3",
        "*r-谷氨酰基转换酶":"feature4",
        "*总蛋白":"feature5",
        "白蛋白":"feature6",
        "*球蛋白":"feature7",
        "白球比例":"feature8",
        "甘油三酯":"feature9",
        "总胆固醇":"feature10",
        "高密度脂蛋白胆固醇":"feature11",
        "低密度脂蛋白胆固醇":"feature12",
        "尿素":"feature13",
        "肌酐":"feature14",
        "尿酸":"feature15",
        "乙肝表面抗原":"feature16",
        "乙肝表面抗体":"feature17",
        "乙肝e抗原":"feature18",
        "乙肝e抗体":"feature19",
        "乙肝核心抗体":"feature20",
        "白细胞计数":"feature21",
        "红细胞计数":"feature22",
        "血红蛋白":"feature23",
        "红细胞压积":"feature24",
        "红细胞平均体积":"feature25",
        "红细胞平均血红蛋白量":"feature26",
        "红细胞平均血红蛋白浓度":"feature27",
        "红细胞体积分布宽度":"feature28",
        "血小板计数":"feature29",
        "血小板平均体积":"feature30",
        "血小板体积分布宽度":"feature31",
        "血小板比积":"feature32",
        "中性粒细胞%":"feature33",
        "淋巴细胞%":"feature34",
        "单核细胞%":"feature35",
        "嗜酸细胞%":"feature36",
        "嗜碱细胞%":"feature37",
        "血糖":"label"
    }, inplace =True)

    for col in data_train.columns:
        data_train = data_train[data_train[col].notnull()]
    return data_train