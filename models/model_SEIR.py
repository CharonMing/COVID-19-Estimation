import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt

# N为人群总数
N = 59170000;#湖北省人口总数
# β为传染率系数
beta = 1
# gamma为恢复率系数
gamma = 0.02
# Te为疾病潜伏期
Te = 14
# I为感染者的初始人数
I = 290
# E为潜伏者的初始人数
E = 1157
# R为治愈者的初始人数
R = 22
# S为易感者的初始人数
S = N - I - E - R
# T为传播时间
T = 150

# INI为初始状态下的数组
INI = (S,E,I,R)


# def funcSEIR(inivalue,_):
#     Y = np.zeros(4)
#     X = inivalue
#     # 易感个体变化
#     Y[0] = - (beta * X[0] * X[2]) / N
#     # 潜伏个体变化
#     Y[1] = (beta * X[0] * X[2]) / N - X[1] / Te
#     # 感染个体变化
#     Y[2] = X[1] / Te - gamma * X[2]
#     # 治愈个体变化
#     Y[3] = gamma * X[2]
#     return Y

# T_range = np.arange(0,T + 1)

# RES = spi.odeint(funcSEIR,INI,T_range)

# plt.rcParams['font.family'] = 'SimHei'
# plt.plot(RES[:,0],color = 'darkblue',label = '易感者',marker = '.')
# plt.plot(RES[:,1],color = 'orange',label = '潜伏者',marker = '.')
# plt.plot(RES[:,2],color = 'red',label = '感染者',marker = '.')
# plt.plot(RES[:,3],color = 'green',label = '康复人群',marker = '.')

# plt.title('SEIR模型对湖北疫情的估计')
# plt.legend()
# plt.xlabel('Day')
# plt.ylabel('Number')
# plt.show()
#参数设置
#模型初值设定
import pandas as pd
data = pd.read_csv('../data/province_2020_07_10.csv',
                   index_col=0, encoding='gb2312')
index = data.index
col = data.columns
class_names = np.unique(data.iloc[:, -1])

S_H = data.iloc[:,0].values.reshape(-1, 1)
print(data,S_H)
S1=59170000;#湖北省人口总数
S2=S3=59170000;
E1=1157;# 1月20至1月2日新增确诊病例数
E2=E3=1157;
I1=599;#感染者
I2=I3=599;
Sq1=2606;#尚在接受医学观察的人数
Sq2=Sq3=2606;
Eq1=300;#估计值，为正在被隔离的潜伏者
Eq2=Eq3=300;
H1=1303;#正在住院的患者，为感染者和被隔离的潜伏者之和
H2=H3=1303;
R1=44;#官方公布的出院人数
R2=44;
R3=44;
#模型参数设定
c=10;#接触率
c1=c;#接触率
c2=c;#接触率
c3=c;#接触率
deltaI=0.13;#感染者的隔离速度
deltaq=0.13;#隔离潜伏者向隔离感染者的转化速率
gammaI=0.007;#感染者的恢复率
gammaH=0.014;#隔离感染者的恢复速率
beta=2.05*10**(-9);#传染概率
q1=1*10**(-6);#隔离比例
q2 = q1
q3 = q1
alpha=2.7*10**(-4);#病死率
rho1=1;#有效接触系数，参考取1
rho2=0.2
rho3=0.1
theta1=1;#潜伏者相对于感染者的传染能力比值
theta2=theta3=1;#不考虑潜伏者传染的情况
lambd=1/14;#隔离接触速度，为14天的倒数
sigma=1/7;#潜伏者向感染者的转化速度，平均潜伏期为7天，为7天的倒数
#差分迭代方程
T=150;

S1_list =[]
S2_list =[]
S3_list =[]
for idx in range(25):
    #有潜伏者传染情况
    S1=S1-(rho1*c1*beta+rho1*c1*q1*(1-beta))*S1*(I1+theta1*E1)+lambd*Sq1;#易感人数迭代
    E1=E1+rho1*c1*beta*(1-q1)*S1*(I1+theta1*E1)-sigma*E1;#潜伏者人数迭代
    I1=I1+sigma*E1-(deltaI+alpha+gammaI)*I1;#感染者人数迭代
    Sq1=Sq1+rho1*c1*q1*(1-beta)*S1*(I1+theta1*E1)-lambd*Sq1;#隔离易感染着人数迭代
    Eq1=Eq1+rho1*c1*beta*q1*S1*(I1+theta1*E1)-deltaq*Eq1;#隔离潜伏者人数迭代
    H1=H1+deltaI*I1+deltaq+Eq1-(alpha+gammaH)*H1;#住院患者人数迭代
    R1=R1+gammaI*I1+gammaH*H1;#康复人数迭代 
    S1_list.append([S1,E1,I1,Sq1,Eq1,H1,R1])
    #不考虑潜伏者传染情况
    S2=S2-(rho2*c2*beta+rho2*c2*q2*(1-beta))*S2*(I2+theta2*E2)+lambd*Sq2;#易感人数迭代
    E2=E2+rho2*c2*beta*(1-q2)*S2*(I2+theta2*E2)-sigma*E2;#潜伏者人数迭代
    I2=I2+sigma*E2-(deltaI+alpha+gammaI)*I2;#感染者人数迭代
    Sq2=Sq2+rho2*c2*q2*(1-beta)*S2*(I2+theta2*E2)-lambd*Sq2;#隔离易感染着人数迭代
    Eq2=Eq2+rho2*c2*beta*q2*S2*(I2+theta2*E2)-deltaq*Eq2;#隔离潜伏者人数迭代
    H2=H2+deltaI*I2+deltaq+Eq2-(alpha+gammaH)*H2;#住院患者人数迭代
    R2=R2+gammaI*I2+gammaH*H2;#康复人数迭代 
    S2_list.append([S2,E2,I2,Sq2,Eq2,H2,R2])
    S3=S3-(rho3*c3*beta+rho3*c3*q3*(1-beta))*S3*(I3+theta3*E3)+lambd*Sq3;#易感人数迭代
    E3=E3+rho3*c3*beta*(1-q3)*S3*(I3+theta2*E3)-sigma*E3;#潜伏者人数迭代
    I3=I3+sigma*E3-(deltaI+alpha+gammaI)*I3;#感染者人数迭代
    Sq3=Sq3+rho3*c3*q3*(1-beta)*S3*(I3+theta3*E3)-lambd*Sq3;#隔离易感染着人数迭代
    Eq3=Eq3+rho3*c3*beta*q3*S3*(I3+theta3*E3)-deltaq*Eq3;#隔离潜伏者人数迭代
    H3=H3+deltaI*I3+deltaq+Eq3-(alpha+gammaH)*H3;#住院患者人数迭代
    R3=R3+gammaI*I3+gammaH*H3;#康复人数迭代 
    S3_list.append([S3,E3,I3,Sq3,Eq3,H3,R3])
plt.rcParams['font.family'] = 'SimHei'
# plt.plot(np.array(S1_list)[:,0],color = 'darkblue',label = 'Susceptible',marker = '.')
# plt.plot(np.array(S1_list)[:,1],color = 'orange',label = 'Exposed',marker = '.')
plt.plot(np.array(S1_list)[:,2],color = 'red',label = '日常防护',marker = '.')
plt.plot(np.array(S2_list)[:,2],color = 'blue',label = '日常防护加强0.2P',marker = '.')
plt.plot(np.array(S3_list)[:,2],color = 'green',label = '日常防护加强0.1p',marker = '.')
plt.plot(np.array(S_H[0:25]),color = 'black',label = '湖北实际情况',marker = '.')
# plt.plot(np.array(S1_list)[:,3],color = 'green',label = 'Recovery',marker = '.')

plt.title('日常防护对湖北疫情的影响')
plt.legend()
plt.xlabel('疫情前期')
plt.ylabel('感染人数')
plt.show()