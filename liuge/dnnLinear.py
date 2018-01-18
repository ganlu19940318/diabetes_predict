import numpy as np
from sklearn.cross_validation import KFold
import tensorflow as tf
from load_data import *
from feature_select import *
from sklearn.metrics import mean_squared_error
FEATURN_NUM = 20
train_data, test_data = preprocessed_data()
importance = load_importance()

FEATURE_COLUMNS = list(importance.index[0:FEATURN_NUM])
print('选用特征：',FEATURE_COLUMNS)
X = train_data[FEATURE_COLUMNS]
y = train_data['血糖']
X_sub = test_data[FEATURE_COLUMNS]
FEATURE_COLUMNS = [str(k) for k in range(len(FEATURE_COLUMNS))]
feature_cols = [tf.contrib.layers.real_valued_column(k) for k in  FEATURE_COLUMNS]
X.columns = FEATURE_COLUMNS
X_sub.columns = FEATURE_COLUMNS
HIDDEN_UNITS = [64, 128]

import time
model_name = 'dnn-f'+str(FEATURN_NUM)+'-'+str(HIDDEN_UNITS)+'-'+time.strftime("%m%d%H%M", time.localtime())
# ## 定义 regressor
config = tf.contrib.learn.RunConfig(gpu_memory_fraction=0.3, log_device_placement=False)
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=HIDDEN_UNITS,
                                          model_dir='./models/'+model_name,
                                          config=config)

# k折交叉
kf = KFold(len(X), n_folds = 6, shuffle=True)
train_preds = np.zeros(X.shape[0])


for m, (train_index, test_index) in enumerate(kf):
    train_X = X.iloc[train_index]
    train_y = y.iloc[train_index]

    test_X = X.iloc[test_index]
    test_y = y.iloc[test_index]
    # ## 定义 input_fn
    # ## 定义 input_fn
    def input_fn(df, label):
        feature_cols = {k: tf.constant(df[k].values) for k in FEATURE_COLUMNS}
        label = tf.constant(label.values)
        return feature_cols,label
    def input_fn2(df):
        feature_cols = {k: tf.constant(df[k].values) for k in FEATURE_COLUMNS}
        return feature_cols

    def train_input_fn():
        '''训练阶段使用的 input_fn'''
        return input_fn(train_X, train_y)


    def test_input_fn():
        '''测试阶段使用的 input_fn'''
        return input_fn2(test_X )

    def sub_input_fn():
        return input_fn2(X_sub)


    # 训练
    regressor.fit(input_fn=train_input_fn,steps=175)
    # 测试
    prediction_value = regressor.predict(input_fn = test_input_fn,as_iterable=False)
    print('fold score:',mean_squared_error(test_y.values, prediction_value) * 0.5)
    #可视化
    ###画图###########################################################################
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20, 3))  # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
    axes = fig.add_subplot(1, 1, 1)
    test_y_sorted = np.sort(np.reshape(test_y.values, (-1,)))
    indices = np.argsort(np.reshape(test_y.values, (-1,)))
    line1, = axes.plot(range(len(prediction_value)), prediction_value[indices], 'b--', label='cnn', linewidth=2)
    # line2,=axes.plot(range(len(gbr_pridict)), gbr_pridict, 'r--',label='优选参数')
    line3, = axes.plot(range(len(test_y_sorted)), test_y_sorted, 'g', label='true y')
    axes.grid()
    fig.tight_layout()
    # plt.legend(handles=[line1, line2,line3])
    plt.legend(handles=[line1, line3])
    plt.title('cnn results')
    plt.show()
sub_pred = regressor.predict(input_fn = sub_input_fn , as_iterable=True)

sub_pred = pd.DataFrame(sub_pred).round(3)
sub_pred.to_csv('sub/' + model_name + '.csv', header=False, index=False, encoding='utf-8')



