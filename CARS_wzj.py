# -*- coding: utf-8 -*-
# @Time    : 2025/04/25 
# @Author  : Zhijing Wu
# @FileName: CARS_wzj.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import copy

def PC_Cross_Validation(X, y, pc, cv):
    '''
        x :光谱矩阵 nxm
        y :浓度阵 （化学值）
        pc:最大主成分数
        cv:交叉验证数量
    return :
        RMSECV:各主成分数对应的RMSECV
        rindex:最佳主成分数
    '''
    kf = KFold(n_splits=cv)
    RMSECV = []
    for i in range(pc):
        RMSE = []
        for train_index, test_index in kf.split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            pls = PLSRegression(n_components=i + 1)
            pls.fit(x_train, y_train)
            y_predict = pls.predict(x_test)
            RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
        RMSE_mean = np.mean(RMSE)
        RMSECV.append(RMSE_mean)
    rindex = np.argmin(RMSECV)
    return RMSECV, rindex

def Cross_Validation(X, y, pc, cv):
    '''
     x :光谱矩阵 nxm
     y :浓度阵 （化学值）
     pc:最大主成分数
     cv:交叉验证数量
     return :
            RMSECV:各主成分数对应的RMSECV
    '''
    kf = KFold(n_splits=cv)
    RMSE = []
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pls = PLSRegression(n_components=pc)
        pls.fit(x_train, y_train)
        y_predict = pls.predict(x_test)
        RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
    RMSE_mean = np.mean(RMSE)
    return RMSE_mean

def CARS_Cloud(X, y, N=50, f=20, cv=10):
    #X：光谱二维矩阵；y：浓度值；N：循环次数（50次）；
    p = 0.8 #校正集所占比
    m, n = X.shape #m:样本数；n: 波长点数
    u = np.power((n/2), (1/(N-1)))#u，k为常数，用于确定每次循环波长数的保留率
    k = (1/(N-1)) * np.log(n/2)
    cal_num = np.round(m * p)#校正集建模的样品数量
    
    b2 = np.arange(n)#初值为0~n-1的一维“波长位置”数组，但在后面循环中b2会被取代
    x = copy.deepcopy(X)#深度拷贝X，保证X始终不变，但是x在循环中会被取代
    D = np.vstack((np.array(b2).reshape(1, -1), X))#(n+1)*m,将b2 放在 X的第0列
    WaveData = []#列表，用于保存下面各次循环保存的波长点位置。

    WaveNum =[]#列表，用于保存下面各次循环保存的波长点数量
    RMSECV = []#列表，保存下面各次循环的REMSEp数值
    r = []#列表，用于保存下面各次循环的波长点的保留率
    
    for i in range(1, N+1):
        r.append(u*np.exp(-1*k*i))#u*np.exp(-1*k*i):各次循环的波长点的保留率,同书P81推导
        wave_num = int(np.round(r[i-1]*n))#wave_num：每次循环保留的波长点的数量，每次循环都会变
        WaveNum = np.hstack((WaveNum, wave_num))#WaveNum转为np数组，用于保存每次循环保留的波长点
        #的数量
        cal_index = np.random.choice    \
            (np.arange(m), size=int(cal_num), replace=False)#从0~m-1中随机取cal_num个数，
        #且不重复
        wave_index = b2[:wave_num].reshape(1, -1)[0]#（wave_num,）,从b2取前wave_num个数据
        #注意每次循环b2都会变化，取上一次循环b2的前wave_num个数值
        xcal = x[np.ix_(list(cal_index), list(wave_index))]#按cal_index随机取样品数的光谱，
        #按wave_index取波长点数
        #xcal = xcal[:,wave_index].reshape(-1,wave_num)
        ycal = y[cal_index]#按cal_index随机取样品数的SSC理化值
        x = x[:, wave_index]#x被取代（按wave_index取wave_num个波长点数）
        D = D[:, wave_index]#D被取代（按wave_index取wave_num个波长点数）
        d = D[0, :].reshape(1,-1)#d用来保存按wave_index取wave_num个波长点数值
        wnum = n - wave_num#wnum为丢弃的波长点的数量
        if wnum > 0:#如果丢弃的波长点的数量大于0，则d原来丢弃波长点的位置补上-1.
            d = np.hstack((d, np.full((1, wnum), -1)))
        if len(WaveData) == 0:#如果是第一次循环，len(WaveData)==0,则将d赋给WaveData（第一行）
            WaveData = d
        else:
            WaveData  = np.vstack((WaveData, d.reshape(1, -1)))#否则，d往下层叠

        if wave_num < f:#如果保留的波长点数量wave_num，小于f(Pcs主成分数，初值20），
            #则f被wave_num取代。
            f = wave_num

        pls = PLSRegression(n_components=f)#创建PLSR模型，主成分数为f
        pls.fit(xcal, ycal)#训练模型
        beta = pls.coef_#获取模型参数（1，n）反映每个波长点对y变量的相关贡献程度
        """
         sklearn的偏最小二乘模型（PLS）的参数coef_表示模型中每个自变量（特征）的系数.

        在PLS模型中，coef_参数是一个数组，包含了每个特征对应的系数。这些系数反映了每个特征对因变量         的影响程度和方向。具体来说，coef_中的每个元素对应一个特征，其值表示该特征对因变量的影响程         度和方向。正值表示正相关，负值表示负相关.但是负相关绝对值很大的波长点位置也被保留了？
        """
        beta = beta.T#（n,1）
        b = np.abs(beta)#模型参数取绝对值

        b2 = np.argsort(-b.squeeze())#按模型参数绝对值大小，从大到小排序，b2返回原数据在beta的
        #位置序号
        coef = copy.deepcopy(beta)#深度拷贝beta
        coeff = coef[b2, :].reshape(len(b2), -1)#coeff返回的是按绝对值大小从大到小排序后的
        
        rmsecv, rindex = PC_Cross_Validation(xcal, ycal, f, cv)#寻找最佳Pcs数值：rindex+1
        RMSECV.append(Cross_Validation(xcal, ycal, rindex+1, cv))#计算最佳Pcs时的RMSEp
        #（cv=10交叉验证）
    #WaveData是一个非常重要参数，50*600，其每一行对应每一次循环保留的波长点位置数据，丢弃位置
    #用-1补足。
    MinIndex = np.argmin(RMSECV)
    Optimal = WaveData[MinIndex, :]#RMSEp最小作为标准选择WaveData的第几行为最终的波长点位置数组
    OptWave = []
    for i in range(Optimal.shape[0]):
        if Optimal[i] != -1:
            OptWave = np.hstack((OptWave,Optimal[i]))#去除为-1的数值
    ii = np.argsort(OptWave)#从小到大排列，np.argsort返回的是原数据在原数组的位置序号
    OptWave = OptWave[ii]#从小到到排列

    fig = plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fonts = 16
    plt.subplot(211)
    plt.xlabel('蒙特卡洛迭代次数', fontsize=fonts)
    plt.ylabel('被选择的波长数量', fontsize=fonts)
    plt.title('最佳迭代次数：' + str(MinIndex) + '次', fontsize=fonts)
    plt.plot(np.arange(N), WaveNum)

    plt.subplot(212)
    plt.xlabel('蒙特卡洛迭代次数', fontsize=fonts)
    plt.ylabel('RMSECV', fontsize=fonts)
    plt.plot(np.arange(N), RMSECV)

    plt.show()

    return OptWave.astype(int)