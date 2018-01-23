import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network  import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

train = pd.read_csv("d_train_20180102_solve.csv", encoding="gbk", header=0)
test = pd.read_csv("d_test_A_20180102_solve.csv", encoding="gbk", header=0)
train_feat = train.drop(['label', 'id'], axis=1)
test_feat = test.drop('id', axis=1)

train_x = train_feat.drop('label_label', axis=1)
train_y = train_feat['label_label']
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=1, stratify=train_y)
print("y_train = 0 is %d" % np.sum(y_train == 0))
print("y_test = 0 is %d" % np.sum(y_test == 0))
print("y_train = 2 is %d" % np.sum(y_train == 2))
print("y_test = 2 is %d" % np.sum(y_test == 2))
print("y_train = 3 is %d" % np.sum(y_train == 3))
print("y_test = 3 is %d" % np.sum(y_test == 3))
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler.fit(X_test)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(activation='relu',
                    alpha=0.05,
                    beta_1=0.9,
                    beta_2=0.999,
                    hidden_layer_sizes=(13, 13, 13),
                    learning_rate_init=0.002,
                    max_iter=500)
mlp.fit(X_train, y_train)
prediction = mlp.predict(X_test)
print("prediction = 0 is %d" % np.sum(prediction == 0))
print("prediction = 2 is %d" % np.sum(prediction == 2))
print("prediction = 3 is %d" % np.sum(prediction == 3))
c = pd.DataFrame(y_test)
c['p'] = prediction
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))
