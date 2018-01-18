import numpy as np
from sklearn.cross_validation import KFold
import tensorflow as tf
from load_data import *
from feature_select import *
from sklearn.decomposition import PCA


# 权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积处理
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1] x_movement、y_movement步长
    # Must have strides[0] = strides[3] = 1 padding='SAME'表示卷积后长宽不变
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# pool 长宽缩小一倍
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')




FEATURE_MAP_WIDTH = 4
train_data, test_data = preprocessed_data()
importance = load_importance()

feature_colums = list(importance.index[0:36])
print('选用特征：',feature_colums)
# X = train_data[[f for f in train_data.columns if not f == '血糖']].values
X = train_data[feature_colums].values
y = train_data['血糖'].values
# X_sub = test_data.values
X_sub =  test_data[feature_colums].values
pca = PCA(n_components=FEATURE_MAP_WIDTH*FEATURE_MAP_WIDTH)
X = pca.fit_transform(X,y)
X_sub = pca.transform(X_sub)
# y_sub = np.reshape(test_data['血糖'].values,(len(test_data),1))

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, FEATURE_MAP_WIDTH*FEATURE_MAP_WIDTH])  # 原始数据的维度：FEATURE_MAP_WIDTH*FEATURE_MAP_WIDTH
ys = tf.placeholder(tf.float32, [None, 1])  # 输出数据为维度：1

keep_prob = tf.placeholder(tf.float32)  # dropout的比例

x_image = tf.reshape(xs, [-1, FEATURE_MAP_WIDTH, FEATURE_MAP_WIDTH, 1])  # 原始数据变成二维图片FEATURE_MAP_WIDTH*FEATURE_MAP_WIDTH
## conv1 layer ##第一卷积层
W_conv1 = weight_variable([2, 2, 1, 32])  # patch 2x2, in size 1, out size 32,每个2*2像素变成32个像素，就是变厚的过程
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size FEATURE_MAP_WIDTHxFEATURE_MAP_WIDTHx32，长宽不变，高度为32的三维图像
# h_pool1 = max_pool_2x2(h_conv1)     # output size 2x2x32 长宽缩小一倍

## conv2 layer ##第二卷积层
W_conv2 = weight_variable([2, 2, 32, 64])  # patch 2x2, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)  # 输入第一层的处理结果 输出shape FEATURE_MAP_WIDTH*FEATURE_MAP_WIDTH*64

## fc1 layer ##  full connection 全连接层
W_fc1 = weight_variable([FEATURE_MAP_WIDTH * FEATURE_MAP_WIDTH * 64, 512])  # 4x4 ，高度为64的三维图片，然后把它拉成512长的一维数组
b_fc1 = bias_variable([512])

h_pool2_flat = tf.reshape(h_conv2, [-1, FEATURE_MAP_WIDTH * FEATURE_MAP_WIDTH * 64])  # 把4*4，高度为64的三维图片拉成一维数组 降维处理
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 把数组中扔掉比例为keep_prob的元素
## fc2 layer ## full connection
W_fc2 = weight_variable([512, 1])  # 512长的一维数组压缩为长度为1的数组
b_fc2 = bias_variable([1])  # 偏置


prediction =  tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

global_step = tf.Variable(0, name='global_step', trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100, 0.96, staircase=True)

train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy,global_step=global_step)




# k折交叉
kf = KFold(len(X), n_folds = 6, shuffle=True)
train_preds = np.zeros(X.shape[0])

ECOPE = 2000


sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
import time
model_name = 'cnn-f'+str(FEATURE_MAP_WIDTH*FEATURE_MAP_WIDTH)+'-'+time.strftime("%m%d%H%M", time.localtime())
save_path = 'models/'+model_name +'/'

for m, (train_index, test_index) in enumerate(kf):
    min_test_loss = np.Inf
    last_test_loss = -np.Inf
    no_dec_count = 0
    over_count = 0
    train_X = X[train_index]
    test_X = X[test_index]
    train_y = np.reshape(y[train_index], (len(train_index), 1))
    test_y = np.reshape(y[test_index],(len(test_index),1))

    for i in range(ECOPE):
        sess.run(train_step, feed_dict={xs: train_X, ys: train_y, keep_prob: 0.7})
        sess.run(global_step)
        if((i+1)%50 ==0):
            train_loss = 0.5 * sess.run(cross_entropy, feed_dict={xs: train_X, ys: train_y, keep_prob: 1.0})
            test_loss = 0.5 * sess.run(cross_entropy, feed_dict={xs: test_X, ys: test_y, keep_prob: 1.0})

            print('fold',m+1,'step',i+1,'train_loss:',train_loss,'test_loss:',test_loss)  # 输出loss值
            if test_loss>last_test_loss:
                over_count+=1
                if(over_count>=4):
                    saver.restore(sess, save_path=saver.last_checkpoints[-1])
                    break
            else:
                over_count = 0
            last_test_loss = test_loss
            if test_loss<min_test_loss:
                min_test_loss=test_loss
                saver.save(sess,save_path=save_path+model_name,global_step=tf.train.global_step(sess,global_step))
                no_dec_count = 0
            else:
                no_dec_count += 1
                if (no_dec_count >= 5):
                    print(no_dec_count * 50, 'steps not learning,restore last ckpt')
                    saver.restore(sess, save_path=saver.last_checkpoints[-1])
                    break
    # 可视化
    prediction_value = sess.run(prediction, feed_dict={xs: test_X, ys: test_y, keep_prob: 1.0})
    ###画图###########################################################################
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(20, 3))  # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
    axes = fig.add_subplot(1, 1, 1)
    test_y_sorted = np.sort(np.reshape(test_y,(-1,)))
    indices =  np.argsort(np.reshape(test_y,(-1,)))
    line1, = axes.plot(range(len(prediction_value)), prediction_value[indices], 'b--', label='cnn', linewidth=2)
    # line2,=axes.plot(range(len(gbr_pridict)), gbr_pridict, 'r--',label='优选参数')
    line3, = axes.plot(range(len(test_y_sorted)), test_y_sorted, 'g', label='true y')
    axes.grid()
    fig.tight_layout()
    # plt.legend(handles=[line1, line2,line3])
    plt.legend(handles=[line1, line3])
    plt.title('cnn results')
    plt.show()
sub_pred = sess.run(prediction, feed_dict={xs: X_sub ,   keep_prob: 1.0})

sub_pred = pd.DataFrame( sub_pred).round(3)
sub_pred.to_csv('sub/'+model_name+'.csv',header=False,index=False,encoding='utf-8')