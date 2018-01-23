# coding=utf-8

import pandas as pd
from dateutil.parser import parse
import numpy as np
from sklearn import preprocessing



# 读取数据，去掉乙肝数据，把性别和日期转换成数字型，改写特征
def loadgoodData(path):
    data_train = pd.read_csv(path+".csv", header=0, encoding="gbk")
    data_train["性别"] = data_train["性别"].apply(lambda x: 1 if x == "男" else 0)
    if path == "d_train_20180102":
        data_train['体检日期'] = (pd.to_datetime(data_train['体检日期']) - parse('2017-09-20')).dt.days
    elif path == "d_test_A_20180102":
        data_train['体检日期'] = (pd.to_datetime(data_train['体检日期']) - parse('2017-10-10')).dt.days
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
    newtrain = data_train.drop(['feature16', 'feature17', 'feature18', 'feature19', 'feature20'], axis=1)
    newtrain.to_csv(path+"_solve.csv", index=False, encoding="utf-8", header=True)



# 给每一个特征划分一个正常范围，空值为0；0-min为1；min-max为2；大于max为3
def feature_label(x, min, max):
    if x > max:
        return 3
    elif x > min:
        return 2
    elif x > 0:
        return 1
    else:
        return 0
def loaddata2(path):
    newtrain = pd.read_csv(path + "_preprocessing.csv", header=0, encoding="gbk")
    newtrain['feature1_label'] = newtrain['feature1'].apply(lambda x: feature_label(x, 15, 40))

    a = newtrain[newtrain['sex'] == 1][['id', 'feature2']]
    a['feature2_label'] = a['feature2'].apply(lambda x: feature_label(x, 9, 51))
    b = newtrain[newtrain['sex'] == 0][['id', 'feature2']]
    b['feature2_label'] = b['feature2'].apply(lambda x: feature_label(x, 8, 41))
    a = pd.concat([a, b], ignore_index=True)[['id', 'feature2_label']]
    newtrain = pd.merge(newtrain, a, on ='id',  how='left')

    a = newtrain[newtrain['sex'] == 1][['id', 'feature3']]
    a['feature3_label'] = a['feature3'].apply(lambda x: feature_label(x, 45, 125))
    b = newtrain[newtrain['sex'] == 0][['id', 'feature3']]
    b['feature3_label'] = b['feature3'].apply(lambda x: feature_label(x, 50, 135))
    a = pd.concat([a,b], ignore_index=True)[['id', 'feature3_label']]
    newtrain = pd.merge(newtrain, a, on ='id',  how='left')

    newtrain['feature4_label'] = newtrain['feature4'].apply(lambda x: feature_label(x, 0, 50))
    newtrain['feature5_label'] = newtrain['feature5'].apply(lambda x: feature_label(x, 60, 85))
    newtrain['feature6_label'] = newtrain['feature6'].apply(lambda x: feature_label(x, 35, 51))
    newtrain['feature7_label'] = newtrain['feature7'].apply(lambda x: feature_label(x, 20, 35))
    newtrain['feature8_label'] = newtrain['feature8'].apply(lambda x: feature_label(x, 1, 2.5))
    newtrain['feature9_label'] = newtrain['feature9'].apply(lambda x: feature_label(x, 0.45, 1.69))
    newtrain['feature10_label'] = newtrain['feature10'].apply(lambda x: feature_label(x, 2.85, 5.69))
    newtrain['feature11_label'] = newtrain['feature11'].apply(lambda x: feature_label(x, 1, 2))
    newtrain['feature12_label'] = newtrain['feature12'].apply(lambda x: feature_label(x, 0.8, 3.12))
    newtrain['feature13_label'] = newtrain['feature13'].apply(lambda x: feature_label(x, 2.8, 7.1))
    newtrain['feature14_label'] = newtrain['feature14'].apply(lambda x: feature_label(x, 44, 106))
    newtrain['feature15_label'] = newtrain['feature15'].apply(lambda x: feature_label(x, 90, 420))
    newtrain['feature21_label'] = newtrain['feature21'].apply(lambda x: feature_label(x, 4, 10))
    newtrain['feature22_label'] = newtrain['feature22'].apply(lambda x: feature_label(x, 3.5, 5.5))
    newtrain['feature23_label'] = newtrain['feature23'].apply(lambda x: feature_label(x, 120, 160))
    newtrain['feature24_label'] = newtrain['feature24'].apply(lambda x: feature_label(x, 0.3, 0.5))
    newtrain['feature25_label'] = newtrain['feature25'].apply(lambda x: feature_label(x, 80, 99))
    newtrain['feature26_label'] = newtrain['feature26'].apply(lambda x: feature_label(x, 26.5, 33.5))
    newtrain['feature27_label'] = newtrain['feature27'].apply(lambda x: feature_label(x, 300, 360))
    newtrain['feature28_label'] = newtrain['feature28'].apply(lambda x: feature_label(x, 10, 15))
    newtrain['feature29_label'] = newtrain['feature29'].apply(lambda x: feature_label(x, 100, 300))
    newtrain['feature30_label'] = newtrain['feature30'].apply(lambda x: feature_label(x, 7, 11))
    newtrain['feature31_label'] = newtrain['feature31'].apply(lambda x: feature_label(x, 9, 18))
    newtrain['feature32_label'] = newtrain['feature32'].apply(lambda x: feature_label(x, 0.093, 0.305))
    newtrain['feature33_label'] = newtrain['feature33'].apply(lambda x: feature_label(x, 43, 76))
    newtrain['feature34_label'] = newtrain['feature34'].apply(lambda x: feature_label(x, 17, 48))
    newtrain['feature35_label'] = newtrain['feature35'].apply(lambda x: feature_label(x, 4, 10))
    newtrain['feature36_label'] = newtrain['feature36'].apply(lambda x: feature_label(x, 0.5, 5))
    newtrain['feature37_label'] = newtrain['feature37'].apply(lambda x: feature_label(x, 0, 1))
    if path == "train":
        newtrain_label = newtrain.pop('label')
        newtrain.insert(60, 'label', newtrain_label)
    newtrain.to_csv(path + "_solve.csv", index=False, encoding="utf-8", header=True)
    return newtrain

    # 数据标准化,连续变量标准化，分类变量ONE-HOT编码
def data_preprocessing(train, test):
    # data_train = loaddata2("train")
    # data_test = loaddata2("test")
    #读取数据并把训练集和测试集合并
    data_train = pd.read_csv(train + "_solve.csv", header=0, encoding="gbk")
    data_test = pd.read_csv(test + "_solve.csv", header=0, encoding="gbk")
    data = pd.concat([data_train, data_test], ignore_index=True, join_axes=[data_train.columns])
    data_sex = data.pop('sex')
    data.insert(59, 'sex', data_sex)
    # 划分连续变量和分类变量
    continuous_data = data.iloc[:, 1:31]
    classified_data = data.iloc[:, 31:60]
    # 将连续变量标准化
    scaler = preprocessing.StandardScaler().fit(continuous_data)
    continuous_data_scaler = scaler.transform(continuous_data)
    # 将分类变量用ONE-HOT编码
    enc = preprocessing.OneHotEncoder()
    enc.fit(classified_data)
    classified_data_enc = enc.transform(classified_data).toarray()
    data_array = np.hstack((continuous_data_scaler, classified_data_enc))
    # 将训练集和测试集分开
    train = data_array[0:5642, :]
    test = data_array[5642:, :]
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    train.insert(0, 'id', data_train['id'])
    train['label'] = data_train['label']
    test.insert(0, 'id', data_test['id'])
    return train, test
























































    # before_eat = newtrain[newtrain['is_eat'] == 0]
    # after_eat = newtrain[newtrain['is_eat'] == 1]
    # before_eat['label_label'] = before_eat['label'].apply(lambda x: feature_label(x, 0, 7))
    # after_eat['label_label'] = after_eat['label'].apply(lambda x: feature_label(x, 0, 11.1))
    # a = before_eat[['label_label','id']]
    # b = after_eat[['label_label','id']]
    # # a = pd.concat([a,b], ignore_index=True)
    # newtrain = pd.merge(newtrain, a, on ='id',  how='left')







    # for col in data_train.columns:
    #     data_train = data_train[data_train[col].notnull()]
    # return data_train