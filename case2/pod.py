import os
import sys
model_dir = os.path.abspath('./model/ML')
data_dir = os.path.abspath('./dataset/T')
sys.path.append(model_dir)
sys.path.append(data_dir)
import scipy.io as sio
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import joblib
import time
from scipy.ndimage import zoom
from model.ML.surrogatemodel import surrogate_model
from sklearn.decomposition import PCA
from pandas.core import base


class ML_POD(object):
    def __init__(self,size,dataset_path,predata_path,level,acc1,acc2,datasize):
        self.level = level
        self.dataset_path = dataset_path
        self.predata_path = predata_path
        self.acc1 = acc1
        self.acc2 = acc2
        self.size = size
        self.datasize = datasize

    def pod(self,path):
        temperature = sio.loadmat(path)['b']
        pca = PCA(n_components=self.acc1)
        coefficient = pca.fit_transform(temperature)
        return coefficient,pca.mean_,pca.components_

    def interp(self,tempture1,tempture2,level):
        ny = int( 64 / ( 2 ** level ) )
        nx = int( 64 / ( 2 ** level ) )
        tempture2 = tempture2.reshape(self.size,ny,nx)
        result = np.empty((self.size,2*ny,2*nx))
        for i,item in enumerate (tempture2):
            result[i] = zoom(item, 2, order=3)
        result = result.reshape(self.size,-1)
        return tempture1-result

    def pod_cal(self,path1,path2,level):
        tempure1 = sio.loadmat(path1)['b']
        tempure2 = sio.loadmat(path2)['b']
        tempure = ML_POD.interp(self,tempure1,tempure2,level)
        pca = PCA(n_components=self.acc2)
        coefficient = pca.fit_transform(tempure)
        return coefficient,pca.mean_,pca.components_

    def predict(self,path,output,level):
        data_input = sio.loadmat(path)['b']
        data_output = output
        res_true, learn_mean, Loss_train, regressor = surrogate_model(data_input, data_output, 'GaussianProcess')
        joblib.dump(regressor,f'./path/POD/{self.datasize}/ExtraTrees_sim'+str(level)+'.pkl') 
        R_value = 1 - np.sum((learn_mean - res_true) ** 2, axis=0) / np.sum((learn_mean - np.mean(res_true, axis=0))**2, axis=0)
        print(R_value)

    def multilevel(self):
        total_mean = []
        total_base = []
        coeff, data_mean, base = ML_POD.pod(self,self.dataset_path+'/T'+str(4)+'.mat')
        total_mean.append(data_mean)
        total_base.append(base)
        ML_POD.predict(self,self.predata_path,coeff,4)
        for i in range (3,4-self.level,-1):
            coeff_cal, data_mean_cal, base_cal = ML_POD.pod_cal(self,self.dataset_path+'/T'+str(i)+'.mat',self.dataset_path+'/T'+str(i+1)+'.mat',i)
            total_mean.append(data_mean_cal)
            total_base.append(base_cal)
            ML_POD.predict(self,self.predata_path,coeff_cal,4-i)
        sio.savemat(f'./dataset/{self.datasize}/POD/mean'+str(self.level),{'b':total_mean})
        sio.savemat(f'./dataset/{self.datasize}/POD/base'+str(self.level),{'b':total_base})
        return total_mean,total_base


class get_new_pre(object):
    def __init__(self,model_path,meanbase_path,pre_data,level):
        self.model_path = model_path
        self.meanbase_path = meanbase_path
        self.pre_data = pre_data
        self.level = level

    def load_model(self):
        model = []
        temp = joblib.load(self.model_path+'/ExtraTrees_sim4.pkl' )
        model.append(temp)
        for i in range(1,self.level):
            temp = joblib.load(self.model_path+'/ExtraTrees_sim'+str(i)+'.pkl' )
            model.append(temp)
        return model
    
    def get_meanbase(self):
        mean = sio.loadmat(self.meanbase_path+'/mean'+str(self.level))['b'][0]
        base = sio.loadmat(self.meanbase_path+'/base'+str(self.level))['b'][0]
        return mean,base

    def interp1(self,tempture,level):
        ny = int( 64 / ( 2 ** 3 ) )
        nx = int( 64 / ( 2 ** 3 ) )
        tempture = tempture.reshape(ny,nx)
        tempture = zoom(tempture, 2**level, order=3)
        return tempture.reshape(1,-1)

    def interp2(self,tempture,level,i):
        ny = int( 64 / ( 2 ** (3-i) ) )
        nx = int( 64 / ( 2 ** (3-i) ) )
        tempture = tempture.reshape(ny,nx)
        tempture = zoom(tempture, 2**(level), order=3)
        return tempture.reshape(1,-1)

    def model_result(self,modeln,new_data,data_mean,base):
        new_data = np.array(new_data).reshape(1,-1)
        result = modeln.predict(new_data).dot(base) + data_mean
        # print(modeln.predict(new_data))
        return result

    def multilevel_result(self,model_total,mean,base):
        nx = int(64 / 2 ** (4-self.level))
        ny = int(64 / 2 ** (4-self.level))
        result = get_new_pre.model_result(self,model_total[0],self.pre_data,mean[0],base[0])
        result = get_new_pre.interp1(self,result,self.level-1)
        for i in range (1,self.level):
            result_cal = get_new_pre.model_result(self,model_total[i],self.pre_data,mean[i],base[i])
            if i != self.level-1 :
                result_cal = get_new_pre.interp2(self,result_cal,self.level-1-i,i)
            result = result + result_cal
        result = result.reshape(ny,nx)
        return result

    def multilevel_resultshow(self,model_total,mean,base):
        plt.figure()
        nx = int(64 / 2 ** (4-self.level))
        ny = int(64 / 2 ** (4-self.level))
        result = get_new_pre.model_result(self,model_total[0],self.pre_data,mean[0],base[0])
        plt.subplot(2,2,1)
        plt.imshow(result.reshape(4,16),cmap='coolwarm')
        result = get_new_pre.interp1(self,result,self.level-1)
        for i in range (1,self.level):
            ny1 = int( 64 / (2 ** (3 - i)))
            nx1 = int( 64 / (2 ** (3 - i)))
            result_cal = get_new_pre.model_result(self,model_total[i],self.pre_data,mean[i],base[i])
            plt.subplot(2,2,i+1)
            plt.imshow(result_cal.reshape(ny1,nx1),cmap='coolwarm')
            cbar=plt.colorbar()
            cbar.set_clim(-2, 2)
            if i != self.level-1 :
                result_cal = get_new_pre.interp2(self,result_cal,self.level-1-i,i)
            result = result + result_cal
        result = result.reshape(ny,nx)
        plt.savefig('result/pod_cal.pdf',bbox_inches='tight')

class singlelevel(object):
    def __init__(self,dataset_path,predata_path,acc,datasize):
        self.dataset_path = dataset_path
        self.predata_path = predata_path
        self.acc = acc
        self.datasize = datasize
    def pod(self):
        tempure = sio.loadmat(self.dataset_path)['b']
        pca = PCA(n_components=self.acc)
        coefficient = pca.fit_transform(tempure)
        return coefficient,pca.mean_,pca.components_
    def predict(self):
        coeff, data_mean, base = singlelevel.pod(self)
        data_input = sio.loadmat(self.predata_path)['b']
        res_true, learn_mean, Loss_train, regressor = surrogate_model(data_input, coeff, 'GaussianProcess')
        joblib.dump(regressor,f'./path/POD/{self.datasize}/ExtraTrees_sim_single.pkl')
        R_value = 1 - np.sum((learn_mean - res_true) ** 2, axis=0) / np.sum((learn_mean - np.mean(res_true, axis=0))**2, axis=0)
        print(R_value)
        sio.savemat(f'./dataset/{self.datasize}/POD/mean_single',{'b':data_mean})
        sio.savemat(f'./dataset/{self.datasize}/POD/base_single',{'b':base})
    def get_meanbase(self):
        mean = sio.loadmat(f'./dataset/{self.datasize}/POD/mean_single')['b']
        base = sio.loadmat(f'./dataset/{self.datasize}/POD/base_single')['b']
        return mean,base
    def get_newpre(self,new_data,data_mean,base):
        modeln = joblib.load(f'./path/POD/{self.datasize}/ExtraTrees_sim_single.pkl')
        new_data = np.array(new_data).reshape(1,-1)
        result = modeln.predict(new_data).dot(base) + data_mean
        # print(modeln.predict(new_data))
        return result



if __name__ == '__main__':
    datasize = 2000
    level = 4
    time_begin = time.time()

    # get_pod_path = ML_POD(datasize,f'./dataset/{datasize}/T/Neuman',f'./dataset/para-{datasize}',level,0.999,0.999,datasize)
    # get_pod_path.multilevel()

    # prediction = get_new_pre(f'path/POD/{datasize}',f'dataset/{datasize}/POD',[-90,90,-40,100,-30,60,-70,80,-60,40,-50,20,-30,-40,-10,20],level)
    # model_total = prediction.load_model()
    # mean, base = prediction.get_meanbase()
    # result = prediction.multilevel_result(model_total,mean,base)
    
    # print(np.max(result))

    single_level = singlelevel(f'./dataset/{datasize}/T/Neuman/T1.mat',f'./dataset/para-{datasize}',0.999,datasize)
    single_level.predict()
    single_mean, single_base = single_level.get_meanbase()
    result = single_level.get_newpre([-90,90,-40,100,-30,60,-70,80,-60,40,-50,20,-30,-40,-10,20],single_mean, single_base)

    time_end = time.time()
    time = time_end - time_begin
    # print(result.reshape(64,256))
    print(np.max(result))
    print(time)
    # plt.imshow(result.reshape(64,256))
    np.savetxt(f'result/time/TimeSL1{datasize}.txt',np.zeros([2,2])+time)

