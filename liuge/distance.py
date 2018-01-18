import pandas as pd
from sklearn.metrics import mean_squared_error
import  os
curr_names = os.listdir('sub/')
best = pd.read_csv('sub/0110_pm_result=0.8004.csv',encoding='utf-8',names=[0])
for name in curr_names:
    curr = pd.read_csv('sub/'+name,encoding='utf-8',names=[0])
    print(name,mean_squared_error(best.values,curr.values)*0.5)
# dnn = pd.read_csv('sub/dnn-f20-[64, 128]-01142006.csv',encoding='utf-8',names=[0])
# cnn = pd.read_csv('sub/cnn-f36-01141125.csv',encoding='utf-8',names=[0])
# sss = pd.read_csv('sub/1-12-pm-sub.csv',encoding='utf-8',names=[0])
# new = pd.DataFrame((best.values+cnn.values+dnn.values+sss.values)/4).round(3)
# new.to_csv('sub/combined-cnn+dnn.csv',header=False,index=False,encoding='utf-8')