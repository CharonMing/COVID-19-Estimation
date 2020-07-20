
"""
Anthor:liu jia ming
Date:2020/7
Theme:AQI Prediction
"""
#-*- coding: utf-8 -*-
from keras.optimizers import Adam
from keras.models import load_model
import  pandas as pd
import  numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from models.Model_CNN_LSTM_ATTENTION import attention_model
import matplotlib.pyplot as plt
from pyecharts import Bar
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back),:]
        dataX.append(a[::-1])
        dataY.append(dataset[i+look_back,:])
    X = np.array(dataX)
    y = np.array(dataY)
    return X, y



# 1.数据预处理
DAY_STEPS = 7
INPUT_DIMS = 1
data = pd.read_csv('./data/China_2020_17.csv',
                   index_col=0, encoding='gb2312')
scaler = preprocessing.MinMaxScaler()
index = data.index
col = data.columns
class_names = np.unique(data.iloc[:, -1])

X = scaler.fit_transform(data.iloc[:,0].values.reshape(-1, 1))
y = scaler.fit_transform(data.iloc[:, 0].values.reshape(-1, 1))
# X = data.iloc[:,0].values.reshape(-1, 1)
# y = data.iloc[:,0].values.reshape(-1, 1)

X_test ,_ = create_dataset(X,DAY_STEPS)
_ , y_test = create_dataset(y,DAY_STEPS)

# 2.训练模型
m=load_model("model_300.h5")

print(y_test.shape)
y_test_pred =scaler.inverse_transform( m.predict(X_test))
print(y_test_pred)
y_test = scaler.inverse_transform(y_test)

# 3.验证评估
bar = Bar()
valList = []
valList.append(r2_score(y_test,y_test_pred))
valList.append(mean_squared_error(y_test, y_test_pred))
valList.append(mean_absolute_error(y_test, y_test_pred))
valList.append(explained_variance_score(y_test, y_test_pred))
valList.append(np.mean(np.abs(y_test_pred - y_test) / y_test)*100)
bar.add('CNN-BliLSTM-ATTENTION测试集评估指标', ['r^2', 'MSE', 'MAE', 'EV', 'MAPE'],
        np.round(valList, 4), is_label_show=True, label_text_color='#000')
bar.render('CNN_BliLSTM-ATTENTION测试集评估指标.html')
print(valList)
# 4.可视化分析
plt.rcParams['font.family'] = 'SimHei'
plt.xlabel('未来天数')
plt.ylabel('累计确诊')
plt.plot(np.array(y_test),
         color='darkorange', label='目标值')
plt.plot(y_test_pred,
         color='navy', label='预测值')
plt.title("LSTM(2) 预测结果")
plt.legend()
plt.show()


