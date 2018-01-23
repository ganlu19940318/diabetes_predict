# 赛题传送门:https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.4472939dOQL3mC&raceId=231638


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor


# 载入数据
df = pd.read_csv('d_train_20180102.csv',header=0,encoding='gbk')
# df.info()显示原始数据属性的基本信息
# df.info()
# df.describe()显示属性的一些统计信息
# df.describe()

# 数据简单分析
# 从基本的数据信息中可以看出数据缺失很严重,但是从数据缺失的具体数值上看似乎存在着一定的规律性.
# 由于赛题中的特征都是各种各样的医学特征,所以猜想每一条记录可能有一种对应的体检套餐.
# 不同的体检套餐造成了缺失值的不同,但是相同的体检套餐具有相同的缺失值.


# 简单预处理
# 这里id是个没有意义的值,所以直接删除就好了.
df = df.drop(['id'],axis=1)

# 对性别,体检日期这种非数值型数据,要转换成数值型数据,由于不存在缺失值,所以这里直接处理.
# 性别采用dummy varibles的方式处理
dummies_df = pd.get_dummies(df['性别'])
# dummies_df.describe()
# dummies_df["??"][dummies_df["??"]==1].describe()
# 这里发现dummies_df里面乱入??,查看数据集发现是因为有一个人的性别是??,大概是做体检的时候忘记写性别了?
# 由于只有1个这样的特例,并且这组数据缺失值不严重,而且考虑到训练集中可能也会存在这样的情况,所以我直接标记为unknown了.
dummies_df = dummies_df.rename(columns={'??':'feature_sex_unknown','男':'feature_sex_male','女':'feature_sex_female'})
df = pd.concat([df,dummies_df],axis=1)

# 对体检日期,我暂时直接拆分成年月日三个特征放进去
df['feature_year'] = df["体检日期"].map(lambda x: int(x.split('/')[2]))
df['feature_month'] = df["体检日期"].map(lambda x: int(x.split('/')[1]))
df['feature_day'] = df["体检日期"].map(lambda x: int(x.split('/')[0]))
df = df.drop(['性别','体检日期'],axis=1)

# year跟其他特征值比起来大太多,这里给它缩小处理.
df["feature_year"] = df["feature_year"]/2017

# 查阅相关资料后发现"乙肝表面抗原","乙肝表面抗体","乙肝e抗原","乙肝e抗体","乙肝核心抗体"这5个特征对血糖没有影响.
# 也做了相关性分析(后面补上code),发现这5个特征对血糖的影响程度也是最低的.
# 所以这里把这5个特征直接delete了.
df = df.drop(["乙肝表面抗原","乙肝表面抗体","乙肝e抗原","乙肝e抗体","乙肝核心抗体"],axis=1)


# 缺失值处理
# 数据的所有特征值都是些医学特征,对这些特征看着简直一脸懵逼.
# 考虑到医学特征之间存在相关性,所以可以用随机森林进行缺失值填补了.
# 选出训练集中无损数据集作为随机森林的训练数据.
train_data = df
for col in train_data.columns:
    train_data = train_data[train_data[col].notnull()]
# 查看完整数据的行数
# train_data.describe()
# 这里发现,只有1300多组数据是完整的.

# 这里由于避免麻烦,我还是选择直接用均值填补法,因为体检报告中,缺失值一般来说是属于正常值的
# 后面补上随机森林处理的相关方法
for col in df.columns:
    df.loc[df[col].isnull(),col] = df[col].mean()

# 特征工程
# 考虑到这些特征都是医学特征,医学特征是存在正常值的范围的,比如白蛋白的范围是40-55.
# 通过查相关资料得到所有医学特征的正常范围,并以此为依据构造新的特征,采用dummy varibles的方式表示
def feature(x,minValue,maxValue):
    if x < minValue:
        return 1
    elif x > maxValue:
        return 3
    else:
        return 2
df["temp_lable0"] = df["*天门冬氨酸氨基转换酶"].apply(lambda x: feature(x,8,40))
df["temp_lable1"] = df["*丙氨酸氨基转换酶"].apply(lambda x: feature(x,5,40))
df["temp_lable2"] = df["*碱性磷酸酶"].apply(lambda x: feature(x,47,185))
df["temp_lable3"] = df["*r-谷氨酰基转换酶"].apply(lambda x: feature(x,7,35))
df["temp_lable4"] = df["*总蛋白"].apply(lambda x: feature(x,65,85))
df["temp_lable5"] = df["白蛋白"].apply(lambda x: feature(x,40,55))
df["temp_lable6"] = df["*球蛋白"].apply(lambda x: feature(x,20,40))
df["temp_lable7"] = df["白球比例"].apply(lambda x: feature(x,1.2,2.4))
df["temp_lable8"] = df["甘油三酯"].apply(lambda x: feature(x,0.56,1.7))
df["temp_lable9"] = df["总胆固醇"].apply(lambda x: feature(x,2.9,5.72))
df["temp_lable10"] = df["高密度脂蛋白胆固醇"].apply(lambda x: feature(x,0.94,2))
df["temp_lable11"] = df["低密度脂蛋白胆固醇"].apply(lambda x: feature(x,1.89,3.1))
df["temp_lable12"] = df["尿素"].apply(lambda x: feature(x,2.9,7.5))
df["temp_lable13"] = df["肌酐"].apply(lambda x: feature(x,44,106))
df["temp_lable14"] = df["尿酸"].apply(lambda x: feature(x,90,420))
df["temp_lable15"] = df["白细胞计数"].apply(lambda x: feature(x,3.5,9.5))
df["temp_lable16"] = df["红细胞计数"].apply(lambda x: feature(x,3.8,5.1))
df["temp_lable17"] = df["血红蛋白"].apply(lambda x: feature(x,115,150))
df["temp_lable18"] = df["红细胞压积"].apply(lambda x: feature(x,35,45))
df["temp_lable19"] = df["红细胞平均体积"].apply(lambda x: feature(x,82,100))
df["temp_lable20"] = df["红细胞平均血红蛋白量"].apply(lambda x: feature(x,27,34))
df["temp_lable21"] = df["红细胞平均血红蛋白浓度"].apply(lambda x: feature(x,316,354))
df["temp_lable22"] = df["红细胞体积分布宽度"].apply(lambda x: feature(x,0,14))
df["temp_lable23"] = df["血小板计数"].apply(lambda x: feature(x,125,350))
df["temp_lable24"] = df["血小板平均体积"].apply(lambda x: feature(x,9,13))
df["temp_lable25"] = df["血小板体积分布宽度"].apply(lambda x: feature(x,9,17))
df["temp_lable26"] = df["血小板比积"].apply(lambda x: feature(x,0.108,0.282))
df["temp_lable27"] = df["中性粒细胞%"].apply(lambda x: feature(x,40,70))
df["temp_lable28"] = df["淋巴细胞%"].apply(lambda x: feature(x,20,50))
df["temp_lable29"] = df["单核细胞%"].apply(lambda x: feature(x,3,10))
df["temp_lable30"] = df["嗜酸细胞%"].apply(lambda x: feature(x,0.4,8))
df["temp_lable31"] = df["嗜碱细胞%"].apply(lambda x: feature(x,0,1))
for i in range(32):
    dummies_df = pd.get_dummies(df["temp_lable"+str(i)])
    dummies_df = dummies_df.rename(
        columns={1: 'feature'+str(i)+'_low', 2: 'feature'+str(i)+'_normal', 3: 'feature'+str(i)+'_high'})
    df = pd.concat([df, dummies_df], axis=1)
    df = df.drop(["temp_lable"+str(i)],axis=1)


# 数据规范化
# 不同特征之间的均值差别还是有点大,所以这里做一下特征缩放,将数据压缩到区间[-1,1].
scaler = preprocessing.StandardScaler()
for col in df.columns:
    if col.find("feature") == -1:
        df[col] = (2*df[col] - df[col].max() - df[col].min())/(df[col].max() - df[col].min())

# 数据保存
df.to_csv("ganlu_d_train_20180102.csv",encoding='gbk',index=None)

