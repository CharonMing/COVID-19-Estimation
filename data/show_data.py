import  pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('./province_2020_07_10.csv',
                   index_col=0, encoding='gb2312')
index = data.index
col = data.columns
class_names = np.unique(data.iloc[:, -1])


x1 = data.iloc[:180,0].values.reshape(-1, 1)
x2= data.iloc[:180,1].values.reshape(-1, 1)
x3 = data.iloc[:180,2].values.reshape(-1, 1)
x4= data.iloc[:180,3].values.reshape(-1, 1)
x5 = data.iloc[:180,4].values.reshape(-1, 1)
x6= data.iloc[:180,5].values.reshape(-1, 1)
plt.rcParams['font.family'] = 'SimHei'
plt.xlabel('日期：2020年1月20日-2020年7月17日')
plt.ylabel('人数')
plt.plot(x1,
         color='darkorange', label='累计确诊')
plt.plot(x2,
         color='navy', label='累计治愈')
plt.plot(x3,
         color='black', label='累计死亡')
plt.plot(x4,
         color='green', label='剩余确诊')
plt.plot(x5,
         color='blue', label='今日确诊')
plt.plot(x6,
         color='navy', label='今日治愈')
plt.title("湖北省疫情数据")

plt.legend()
plt.show()