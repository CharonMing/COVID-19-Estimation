
"""
Anthor:liu jia ming
Date:2020/7
Theme:AQI Prediction
"""
#-*- coding: utf-8 -*-
from keras.optimizers import Adam
from keras.models import load_model
from keras.models import Sequential

from keras.layers import Dense, LSTM, BatchNormalization
import  pandas as pd
import  numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from models.Model_CNN_LSTM_ATTENTION import attention_model
import matplotlib.pyplot as plt
from pyecharts import Bar
import keras
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        #创建一个图
        plt.figure()
        # acc
        # plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')#plt.plot(x,y)，这个将数据画成曲线
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            # plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)#设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('loss')#给x，y轴加注释
        plt.legend(loc="upper right")#设置图例显示位置
        plt.show()

#创建一个实例LossHistory
history = LossHistory()

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
lstm_units = 64
data = pd.read_csv('./data/China_2020_07_10.csv',
                   index_col=0, encoding='gb2312')
scaler = preprocessing.MinMaxScaler()
index = data.index
col = data.columns
class_names = np.unique(data.iloc[:, -1])

X = scaler.fit_transform(data.iloc[:,0].values.reshape(-1, 1))
y = scaler.fit_transform(data.iloc[:, 0].values.reshape(-1, 1))
# X = data.iloc[:,0].values.reshape(-1, 1)
# y = data.iloc[:,0].values.reshape(-1, 1)


X_train ,_ = create_dataset(X,DAY_STEPS)
_ , y_train = create_dataset(y,DAY_STEPS)
print(X_train.shape,y_train.shape)
# X_train, X_test, y_train, y_test = X[0:len(X)-12],X[len(X)-12:],y[0:len(X)-12],y[len(X)-12:]

# 2.训练模型

m = attention_model(DAY_STEPS,INPUT_DIMS,lstm_units)
m.summary()
m.compile(optimizer=Adam(lr=0.001), loss='mse')
m.fit(X_train, y_train, epochs=200, batch_size=1, validation_split=0.1,callbacks=[history])

history.loss_plot('epoch')

m.save("./model_LSTM_ATTENTION_600.h5")
