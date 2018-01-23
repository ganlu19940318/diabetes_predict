import pandas as pd
import loaddata as ld
from sklearn.ensemble import RandomForestRegressor

# 用随机森林的方法补充缺失值；用前面列的数据补充后面，用后面列的数据补充前面
def preprocessing(train_path, test_path):
    # ld.loadgoodData("d_train_20180102")
    # 读取数据
    train_feat = pd.read_csv(train_path, encoding="gbk", header=0)
    test_feat = pd.read_csv(test_path, encoding="gbk", header=0)
    cols = train_feat.columns
    # 将训练集和测试集合并到一起
    feat = pd.concat([train_feat, test_feat], axis=0)
    feat = feat[cols]
    # 去掉血糖标签
    feat_label = feat['label']
    feat = feat.drop('label', axis=1)
    # feat = feat.drop(['feature24', 'feature25', 'feature12', 'feature29'], axis=1)
    feature = cols[4:35]
    feature_1 = feature[0:15]
    feature_2 = feature[15:]

    for i in feature_1:
        predicted = set_missing_1(feat, i)
        feat.loc[(feat[i].isnull()), i] = predicted
    for i in feature_2:
        predicted = set_missing_2(feat, i)
        feat.loc[(feat[i].isnull()), i] = predicted
    # 将训练集和测试集分开
    feat = feat.sort_index(axis=0, ascending=True, by='id')
    feat['label'] = feat_label
    train_preprocessing = feat.iloc[0:5642]
    test_preprocessing = feat.iloc[5642:]
    #保存
    train_preprocessing.to_csv("train_preprocessing.csv", index=False, encoding="utf-8", header=True)
    test_preprocessing.to_csv("test_preprocessing.csv", index=False, encoding="utf-8", header=True)

def set_missing_1(df, f):
    known = df.dropna(axis=0, how='any')
    X = known.ix[:, 'feature21':'feature37']
    X = X.as_matrix()
    y = known[f]
    y = y.as_matrix()
    unknown = df[df[f].isnull()]
    unknown = unknown.ix[:, 'feature21':'feature37']
    unknown = unknown.as_matrix()
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, n_jobs=-1)
    rfr.fit(X, y)
    predicted = rfr.predict(unknown)
    return predicted
def set_missing_2(df, f):
    known = df.dropna(axis=0, how='any')
    X = known.ix[:, 'feature1':'feature15']
    X = X.as_matrix()
    y = known[f]
    y = y.as_matrix()
    unknown = df[df[f].isnull()]
    unknown = unknown.ix[:, 'feature1':'feature15']
    unknown = unknown.as_matrix()
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, n_jobs=-1)
    rfr.fit(X, y)
    predicted = rfr.predict(unknown)
    return predicted


# train_feat = train.drop('id', axis=1)
# train_feat.insert(0, 'id', train['id'])


