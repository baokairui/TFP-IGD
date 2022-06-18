import numpy as np
from mlp import MLP
import torch
import scipy.io as scio
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern, RationalQuadratic
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import os
import math

def Z_ScoreNormalization(x, mu, sigma):
    x = (x - mu)/sigma
    return x

def surrogate_model(indata, outdata, regress):
    '''
    封装的函数，输入为:数据对（indata, outdata），向量形式，
                模型regressor，string形式，在六个模型'MLP', 'GaussianProcess', 'DecisionTree',
                   'RandomForest‘, ’ExtraTrees‘ ，’RBF'里任选一个
          输出为：训练过程误差Loss，测试集的真值y_true、预测值y_pred
    '''

    # 剖分训练集和测试集
    np.random.seed(100)
    per = 0.7       #设置训练集采样比例，默认为所有数据的70%，因此测试集为30%
   
    samples_per = math.floor(int(indata.shape[0]*per))  
    sample_choice = np.random.choice(indata.shape[0], size=indata.shape[0], replace=False) #随机排序，随机选取70%
   
    train_data_input = indata[sample_choice[:samples_per], :] 
    train_data_output = outdata[sample_choice[:samples_per], :]
    test_data_input = indata[sample_choice[samples_per:], :]
    test_data_output = outdata[sample_choice[samples_per:], :]
    
    # 无归一化   
    train_input = train_data_input
    train_output = train_data_output
    test_input = test_data_input
    test_output = test_data_output
   
    # 数据归一化Z-score标准化
    # train_input = Z_ScoreNormalization(train_data_input,
    #                                    np.average(train_data_input, axis=0),
    #                                    np.std(train_data_input, axis=0))
    # train_output = Z_ScoreNormalization(train_data_output,
    #                                     np.average(train_data_output, axis=0),
    #                                     np.std(train_data_output, axis=0))
    # test_input = Z_ScoreNormalization(test_data_input,
    #                                   np.average(train_data_input, axis=0),
    #                                   np.std(train_data_input, axis=0)) #由于不具备测试集信息，此处用训练集的均值和方差进行归一化
    # test_output = Z_ScoreNormalization(test_data_output,
    #                                    np.average(train_data_output, axis=0),
    #                                    np.std(train_data_output, axis=0)) #由于不具备测试集信息，此处用训练集的均值和方差进行归一化
  
    # 选择模型进行代理模型构建
    # MLP模型，才有训练过程和训练误差，调用mlp.py文件
    test_pred = None
    if regress == 'MLP':
        lrs = [0.1]
        for i, eta in enumerate(lrs):
            model_mean = MLP(train_input, train_output, nhidden=100)
            model_mean.mlptrain(train_input, train_output, eta=eta, niterations=500000)
            ## 因为训练时间比较久，因此保存神经网络，直接调用就行
            torch.save(model_mean, 'model/net1117_minmax.pkl')           # 保存整个神经网络的结构和模型参数,需要设置路径
            model_mean = torch.load('model/net1117_minmax.pkl')          # 读取神经网络模型参数，需要设置路径
            ## 给定输入，预测输出
            test = np.concatenate((test_input, -np.ones((np.array(test_input).shape[0], 1))), axis=1)
            test_pred = model_mean.mlpfwd(test)
            Loss_train = model_mean.loss
    # 高斯过程模型
    elif regress == 'GaussianProcess':
        kernel = RationalQuadratic()
        regressor = GaussianProcessRegressor(kernel=kernel, random_state=0)
        regressor.fit(train_input, train_output)
        test_pred = regressor.predict(test_input)
        Loss_train = None
    # 决策树模型
    elif regress == 'DecisionTree':
        regressor = DecisionTreeRegressor()
        regressor.fit(train_input, train_output)
        test_pred = regressor.predict(test_input)
        Loss_train = None
    # 随机森林模型
    elif regress == 'RandomForest':
        regressor = RandomForestRegressor(n_estimators=10000, random_state=50)
        regressor.fit(train_input, train_output)
        test_pred = regressor.predict(test_input)
        Loss_train = None
    # 多输出随机森林模型
    elif regress == 'multiRandomForest':
        regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators=500, random_state=50))
        regressor.fit(train_input, train_output)
        test_pred = regressor.predict(test_input)
        Loss_train = None
    # 端随机树模型
    elif regress == 'ExtraTrees':
        regressor = ExtraTreesRegressor()
        regressor.fit(train_input, train_output)
        test_pred = regressor.predict(test_input)
        Loss_train = None
    # 多输出端随机树模型
    elif regress == 'multiExtraTrees':
        regressor = MultiOutputRegressor(ExtraTreesRegressor())
        regressor.fit(train_input, train_output)
        test_pred = regressor.predict(test_input)
        Loss_train = None
    # RBF模型
    elif regress == 'RBF':
        test_pred = np.zeros_like(test_output)
        regressor = SVR(kernel='rbf')
        for feature in range(np.array(test_output).shape[1]):
            regressor.fit(train_input, train_output[:,feature])
            test_pred[:,feature] = regressor.predict(test_input)
        Loss_train = None

    # ## 数据去归一化Z-score
    # test_pred = np.average(train_data_output, axis=0) + \
    #              np.std(train_data_output, axis=0)*test_pred
    test_true = np.array(test_data_output)
    test_pred = np.reshape(test_pred,(test_true.shape[0],test_true.shape[1]))
    return test_true, test_pred, Loss_train, regressor