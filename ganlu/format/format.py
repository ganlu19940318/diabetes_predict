# 赛题传送门:https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.4472939dOQL3mC&raceId=231638


# 赛题答疑汇总
# 1.糖化血糖蛋白跟血糖相关性极高(然而体检没有这个特征) # nothing to do #
# 2.体检数据是空腹体检,心情好的数据(所以不用考虑心情,天气,日期等因素) # delete date #
# 3.性别与糖尿病没关系 # delete sex #
# 4.年龄跟血糖有一定的相关性 # 40-59是高发人群 #
# 5.肝功能受到损害会影响血糖 # 高斯混合模型聚类 #
# 6.理论上,各个医院的检测结果通用 # nothing to do #
# 7.白细胞，嗜酸碱细胞与血糖无相关 # delete all #
# 8.血脂,蛋白质,氨基酸,血压与血糖最相关
# 9.携带乙肝影响不大，肝功异常，转氨酶升高，影响血糖 # delete乙肝 #
# 10.红细胞分布,红细胞体积与血糖无关 # delete all#
# 11.血糖和尿酸(肾功能)存在相关性,糖尿病到一定程度会影响肾功能 # 添加额外一个特征标记高尿酸人群 #
# 12.AB榜的数据分布基本一致 # nothing to do #

import math
from sklearn.mixture import GMM
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor


# 载入数据
df = pd.read_csv('d_train_20180102.csv',header=0,encoding='gbk')
# df = pd.read_csv('d_test_A_20180102.csv',header=0,encoding='gbk')
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

# 专家答疑后指出性别和糖尿病无关,所以把性别delete
if 'feature_sex_unknown' in df.columns:
    df = df.drop(['feature_sex_unknown'], axis=1)
df = df.drop(['feature_sex_male','feature_sex_female'],axis=1)

# 对体检日期,我暂时直接拆分成年月日三个特征放进去
df['feature_year'] = df["体检日期"].map(lambda x: int(x.split('/')[2]))
df['feature_month'] = df["体检日期"].map(lambda x: int(x.split('/')[1]))
df['feature_day'] = df["体检日期"].map(lambda x: int(x.split('/')[0]))
df = df.drop(['性别','体检日期'],axis=1)


# year跟其他特征值比起来大太多,这里给它缩小处理.
df["feature_year"] = df["feature_year"]/2017
# 专家答疑后指出日期和糖尿病无关,所以把日期delete
df = df.drop(['feature_year','feature_month','feature_day'],axis=1)

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
df["temp_lable5"] = df["白蛋白"].apply(lambda x: feature(x,35,55))
df["temp_lable6"] = df["*球蛋白"].apply(lambda x: feature(x,20,35))
df["temp_lable7"] = df["白球比例"].apply(lambda x: feature(x,1.2,2.4))
df["temp_lable8"] = df["甘油三酯"].apply(lambda x: feature(x,0.56,1.7))
df["temp_lable9"] = df["总胆固醇"].apply(lambda x: feature(x,2.9,5.72))
df["temp_lable10"] = df["高密度脂蛋白胆固醇"].apply(lambda x: feature(x,0.94,2))
df["temp_lable11"] = df["低密度脂蛋白胆固醇"].apply(lambda x: feature(x,1.89,3.1))
df["temp_lable12"] = df["尿素"].apply(lambda x: feature(x,2.9,7.5))
df["temp_lable13"] = df["肌酐"].apply(lambda x: feature(x,44,106))
df["temp_lable14"] = df["尿酸"].apply(lambda x: feature(x,90,420))
df["temp_lable15"] = df["白细胞计数"].apply(lambda x: feature(x,4,10))
df["temp_lable16"] = df["红细胞计数"].apply(lambda x: feature(x,3.5,5.5))
df["temp_lable17"] = df["血红蛋白"].apply(lambda x: feature(x,110,160))
df["temp_lable18"] = df["红细胞压积"].apply(lambda x: feature(x,36,50))
df["temp_lable19"] = df["红细胞平均体积"].apply(lambda x: feature(x,86,100))
df["temp_lable20"] = df["红细胞平均血红蛋白量"].apply(lambda x: feature(x,26,31))
df["temp_lable21"] = df["红细胞平均血红蛋白浓度"].apply(lambda x: feature(x,316,354))
df["temp_lable22"] = df["红细胞体积分布宽度"].apply(lambda x: feature(x,11,16))
df["temp_lable23"] = df["血小板计数"].apply(lambda x: feature(x,100,300))
df["temp_lable24"] = df["血小板平均体积"].apply(lambda x: feature(x,9,13))
df["temp_lable25"] = df["血小板体积分布宽度"].apply(lambda x: feature(x,9,17))
df["temp_lable26"] = df["血小板比积"].apply(lambda x: feature(x,0.108,0.282))
df["temp_lable27"] = df["中性粒细胞%"].apply(lambda x: feature(x,45,77))
df["temp_lable28"] = df["淋巴细胞%"].apply(lambda x: feature(x,20,40))
df["temp_lable29"] = df["单核细胞%"].apply(lambda x: feature(x,3,10))
df["temp_lable30"] = df["嗜酸细胞%"].apply(lambda x: feature(x,0.5,5))
df["temp_lable31"] = df["嗜碱细胞%"].apply(lambda x: feature(x,0,1))
for i in range(32):
    dummies_df = pd.get_dummies(df["temp_lable"+str(i)])
    dummies_df = dummies_df.rename(
        columns={1: 'feature'+str(i)+'_low', 2: 'feature'+str(i)+'_normal', 3: 'feature'+str(i)+'_high'})
    df = pd.concat([df, dummies_df], axis=1)
    df = df.drop(["temp_lable"+str(i)],axis=1)
# 专家答疑后指出白细胞,嗜酸碱细胞和糖尿病无关,所以把白细胞,嗜酸碱细胞delete
# 专家答疑后指出红细胞分布,红细胞体积和糖尿病无关,所以把红细胞分布,红细胞体积delete
df = df.drop(["白细胞计数","feature15_low","feature15_normal","feature15_high",
              "嗜酸细胞%","feature30_low", "feature30_normal", "feature30_high",
              "嗜碱细胞%","feature31_normal", "feature31_high",
              "红细胞平均体积","feature19_low", "feature19_normal", "feature19_high",
              "红细胞体积分布宽度","feature22_low", "feature22_normal", "feature22_high"],axis=1)

# 年龄跟血糖有一定的相关性,在中国糖尿病高发年龄大约在40~59之间
df["age_feature"] = df["年龄"].apply(lambda x: feature(x,40,59))

# 从这里开始大量造特征
numerics = df.loc[:, ["年龄","*天门冬氨酸氨基转换酶","*丙氨酸氨基转换酶","*碱性磷酸酶",
                      "*r-谷氨酰基转换酶","*总蛋白","白蛋白","*球蛋白","白球比例",
                      "甘油三酯","总胆固醇","高密度脂蛋白胆固醇","低密度脂蛋白胆固醇",
                      "尿素","肌酐","尿酸","红细胞计数","血红蛋白","红细胞压积",
                      "红细胞平均血红蛋白量","红细胞平均血红蛋白浓度","血小板计数",
                      "血小板平均体积","血小板体积分布宽度","血小板比积","中性粒细胞%",
                      "淋巴细胞%","单核细胞%"]]
for i in range(0, numerics.columns.size - 1):
    for j in range(0, numerics.columns.size - 1):
        print(str(i) + "," + str(j) + ":" + "working")
        if i <= j:
            name = str(numerics.columns.values[i]) + "*" + str(numerics.columns.values[j])
            df = pd.concat([df, pd.Series(numerics.iloc[:, i] * numerics.iloc[:, j], name=name)], axis=1)
        if i < j:
            name = str(numerics.columns.values[i]) + "+" + str(numerics.columns.values[j])
            df = pd.concat([df, pd.Series(numerics.iloc[:, i] + numerics.iloc[:, j], name=name)], axis=1)
        if not i == j:
            name = str(numerics.columns.values[i]) + "/" + str(numerics.columns.values[j])
            df = pd.concat([df, pd.Series(numerics.iloc[:, i] / numerics.iloc[:, j], name=name)], axis=1)
            name = str(numerics.columns.values[i]) + "-" + str(numerics.columns.values[j])
            df = pd.concat([df, pd.Series(numerics.iloc[:, i] - numerics.iloc[:, j], name=name)], axis=1)
        if i == j:
            name = "log" + str(numerics.columns.values[i])
            df = pd.concat([df, pd.Series(numerics.iloc[:, i], name=name).apply(lambda x: math.log(x))], axis=1)
            name = "sqrt" + str(numerics.columns.values[i])
            df = pd.concat([df, pd.Series(numerics.iloc[:, i], name=name).apply(lambda x: math.sqrt(x))], axis=1)
            name = "log(sqrt)" + str(numerics.columns.values[i])
            df = pd.concat([df, pd.Series(numerics.iloc[:, i], name=name).apply(lambda x: math.log(math.sqrt(x)))], axis=1)


# 肝功能受到损害会影响血糖
# 传送门:http://health.sina.com.cn/zl/d/sbjy/2014-12-16/1713139.shtml
# 肝功能异常指标
# 针对已有特征,选择"*天门冬氨酸氨基转换酶","*丙氨酸氨基转换酶","*碱性磷酸酶","*r-谷氨酰基转换酶","白蛋白","*球蛋白"作为特征
# 通过上面的特征,做聚类分出肝功能是否异常
def temp_feature(x,Value):
    if x < Value:
        return 0
    else:
        return 1
temp_df = df[["*天门冬氨酸氨基转换酶","*丙氨酸氨基转换酶","*碱性磷酸酶","*r-谷氨酰基转换酶","白蛋白","*球蛋白"]]
temp_df["temp_lable0"] = temp_df.loc[:,"*天门冬氨酸氨基转换酶"].apply(lambda x: temp_feature(x,40))
temp_df["temp_lable1"] = temp_df.loc[:,"*丙氨酸氨基转换酶"].apply(lambda x: temp_feature(x,40))
temp_df["temp_lable2"] = temp_df.loc[:,"*碱性磷酸酶"].apply(lambda x: temp_feature(x,185))
temp_df["temp_lable3"] = temp_df.loc[:,"*r-谷氨酰基转换酶"].apply(lambda x: temp_feature(x,35))
temp_df["temp_lable4"] = temp_df.loc[:,"白蛋白"].apply(lambda x: temp_feature(x,55))
temp_df["temp_lable5"] = temp_df.loc[:,"*球蛋白"].apply(lambda x: temp_feature(x,35))
gmm = GMM(n_components=2).fit(temp_df)
labels = gmm.predict(temp_df)
df["liver_trouble_feature"] = labels

# 专家指出,血糖和尿酸(肾功能)存在相关性,糖尿病到一定程度会影响肾功能
# 于是,这里额外添加一个特征标记高尿酸的人群
def temp_feature(x,Value):
    if x < Value:
        return 0
    else:
        return 1
df["high_feature_UA"] = df["尿酸"].apply(lambda x: temp_feature(x,420))

# 数据规范化
# 不同特征之间的均值差别还是有点大,所以这里做一下特征缩放,将数据压缩到区间[-1,1].
# 只对特征名中不含有feature的特征处理
# 只对特征名中不含有"血糖"的特征处理
for col in df.columns:
    if col.find("feature") == -1 \
            and col.find("血糖") == -1:
        df[col] = (2*df[col] - df[col].max() - df[col].min())/(df[col].max() - df[col].min())

# 把血糖挪到最后一列
if "血糖" in df.columns:
    df_temp = df.pop("血糖")
    df = pd.concat([df,df_temp],axis=1)
# 数据保存
print("****************************************data saving****************************************")
df.to_csv("ganlu_d_train_20180102.csv",encoding='gbk',index=None)
# df.to_csv("ganlu_d_test_20180102.csv",encoding='gbk',index=None)
print("****************************************all good****************************************")
