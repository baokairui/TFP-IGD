import os
import sys
model_dir = os.path.abspath('./model/FNO')
sys.path.append(model_dir)
import scipy.io as sio
import numpy as np
import torch
from pod import get_new_pre, singlelevel
from model.Unet.unet import UNetV2
from model.FNO.fourier_2d import FNO2d
from scipy.interpolate import Rbf
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def inter_tem(lmx,lmy,pmx,pmy,t):
    func = Rbf(lmx.reshape(-1), lmy.reshape(-1), t, function='linear')
    data_t = func(pmx,pmy)
    data_t = data_t.reshape(64,64)
    return data_t

def get_result(output, data):
        result = data.clone()
        result[...,1:-1,1:-1] = output[...,1:-1,1:-1]
        return result

def get_data(data1,data2):
    ny = data1[0].shape[0]
    nx = data1[0].shape[1]
    data = np.zeros((data1.shape[0],2,ny,nx))
    for i in range(len(data)):
        data[i][0] = data1[i]
        data[i][1] = data2[i]
    return data

def get_data1(data1,data2):
        data = np.zeros((1,2,64,64))
        data[0][0] = data1
        data[0][1] = data2
        return data

def get_result_norm(net,data):
    data_input = torch.empty_like(data)
        # x = torch.max(data[0][0])
        # y = torch.max(data[0][1])
    x = 100
    y = 90
        # print(x,y)
    data_input[0][0] = data[0][0] / x
    data_input[0][1] = data[0][1] / y
        # print(data_input[0][1])
        # net.load_state_dict(torch.load("path/DeepLearning/best_mesh.pth"))
    data_output = net(data_input)
        # print(data_output[0][1])
        # output = torch.empty_like(data_output)
    data_output[0][0] = data_output[0][0] * x
    data_output[0][1] = data_output[0][1] * y
    result = data.clone()
    result[...,1:-1,1:-1] = data_output[...,1:-1,1:-1]
    return result


# meshx = sio.loadmat('./dataset/2000/meshx/meshx1')['b']
# meshy = sio.loadmat('./dataset/2000/meshy/meshy1')['b']
# boundx = sio.loadmat('./dataset/2000/bound/boundx1')['b']
# boundy = sio.loadmat('./dataset/2000/bound/boundy1')['b']
# net1 = UNetV2(in_channels=2, num_classes=2).to(device)
# net1.load_state_dict(torch.load("path/Mesh/best_mesh_phyaccnorm.pth"))


# datasize = 2000
# prediction = get_new_pre(f'path/POD/{datasize}',f'dataset/{datasize}/POD',[-90,90,-40,100,-30,60,-70,80,-60,40,-50,20,-30,-40,-10,20],4)
# model_total = prediction.load_model()
# mean, base = prediction.get_meanbase()
# label = sio.loadmat(f'dataset/{datasize}/T/Neuman/T1.mat')['b'].reshape(datasize,64,64)
# data = sio.loadmat(f'dataset/para-{datasize}.mat')['b']
# mae_ml = 0
# mre_ml = 0
# temp = np.array([])
# for i,item in enumerate(data):
#     print(i)
#     prediction1 = get_new_pre(f'path/POD/{datasize}',f'dataset/{datasize}/POD',item,4)
#     ml = prediction1.multilevel_result(model_total,mean,base)
#     error = np.abs(ml-label[i])
#     mae_ml = mae_ml + np.mean(error)
#     mre_ml = mre_ml + np.mean(error/label[i])
#     temp = np.append(temp,np.max(error))
#     # print(mae_ml, mre_ml)

# print(mae_ml/data.shape[0], mre_ml/data.shape[0])
# print(np.mean(temp))

# prediction = get_new_pre('path/POD/2000','dataset/2000/POD',[-90,90,-40,100,-30,60,-70,80,-60,40,-50,20,-30,-40,-10,20],4)
# model_total = prediction.load_model()
# mean, base = prediction.get_meanbase()
# label = sio.loadmat('dataset/2000/T/Neuman/T1.mat')['b'].reshape(2000,64,64)
# data = sio.loadmat('dataset/para-2000')['b']
# mae_ml = 0
# mre_ml = 0
# mae_int = 0
# mre_int = 0
# # error_single1 = 0
# for i,item in enumerate(data):
#     print(i)

#     F = get_data1(boundx[i],boundy[i])
#     F = torch.from_numpy(F).type(torch.FloatTensor).to(device)
#     with torch.no_grad():
#         y1 = get_result_norm(net1,F)

#     prediction1 = get_new_pre('path/POD/2000','dataset/2000/POD',item,4)
#     ml = prediction1.multilevel_result(model_total,mean,base)
#     error = np.abs(ml-label[i])

#     tem = inter_tem(meshx[i],meshy[i],y1[0][0].cpu().numpy(),y1[0][1].cpu().numpy(),ml)
#     error1 = np.abs(tem-label[i])
#     mae_int = mae_int + np.mean(error1)
#     mre_int = mre_int + np.mean(error1/label[i])
#     # print(np.mean(error))
#     mae_ml = mae_ml + np.mean(error)
#     mre_ml = mre_ml + np.mean(error/label[i])
#     print(mae_ml, mre_ml)
#     print(mae_int, mre_int)
# print(mae_ml/data.shape[0], mre_ml/data.shape[0])
# print(mae_int/data.shape[0], mre_int/data.shape[0])

datasize = 2000

# prediction = singlelevel(f'./dataset/{datasize}/T/Neuman/T1.mat',f'dataset/para-{datasize}',0.999,datasize)
# mean, base = prediction.get_meanbase()
# label = sio.loadmat(f'dataset/{datasize}/T/Neuman/T1.mat')['b'].reshape(datasize,64,64)
# data = sio.loadmat(f'dataset/para-{datasize}')['b']
# mae_ml = 0
# mre_ml = 0
# for i,item in enumerate(data):
#     print(i)
#     ml = prediction.get_newpre(item,mean,base).reshape(64,64)
#     error = np.abs(ml-label[i])
#     mae_ml = mae_ml + np.mean(error)
#     mre_ml = mre_ml + np.mean(error/label[i])
# print(mae_ml/data.shape[0], mre_ml/data.shape[0])

# bx = sio.loadmat(f'./dataset/{datasize}/bound/boundx1')['b']
# by = sio.loadmat(f'./dataset/{datasize}/bound/boundy1')['b']
# bx = sio.loadmat(f'./dataset/{datasize}/meshx/meshx1')['b']
# by = sio.loadmat(f'./dataset/{datasize}/meshy/meshy1')['b']
# label = sio.loadmat(f'dataset/{datasize}/T/Neuman/T1.mat')['b'].reshape(datasize,64,64)
# data_input = get_data(bx, by)
# F = torch.from_numpy(data_input).type(torch.FloatTensor).to(device)
# net1 = FNO2d(12, 12, 32, input_channels=2).to(device)
# net2 = UNetV2(in_channels=2, num_classes=1).to(device)
# net1.load_state_dict(torch.load(f"path/Thermal/irr-fno-{datasize}full.pth"))
# net2.load_state_dict(torch.load(f"path/Thermal/irr-unet-{datasize}full.pth"))
# mae_unet = 0
# mae_fno = 0
# mre_unet = 0
# mre_fno = 0

# for i,item in enumerate(data_input):
#     print(i)
#     F = torch.from_numpy(item).type(torch.FloatTensor).to(device).reshape(1,2,64,64)
#     with torch.no_grad():  
#         fno_p = net1(F)
#         unet_p = net2(F)
#         # print(unet_p.shape)
#     fno = fno_p[0][0].cpu().numpy()
#     unet = unet_p[0][0].cpu().numpy()
#     error_fno = np.abs(fno-label[i])
#     error_unet = np.abs(unet-label[i])
#     mae_unet = mae_unet + np.mean(error_unet)
#     mae_fno = mae_fno + np.mean(error_fno)
#     mre_unet = mre_unet + np.mean(error_unet/label[i])
#     mre_fno = mre_fno + np.mean(error_fno/label[i])
# print(mae_unet/label.shape[0],mre_unet/label.shape[0])
# print(mae_fno/label.shape[0],mre_fno/label.shape[0])

bx = sio.loadmat('./dataset/2000/bound/boundx1')['b']
by = sio.loadmat('./dataset/2000/bound/boundy1')['b']
labelx = sio.loadmat('./dataset/2000/meshx/meshx1')['b'].reshape(2000,64,64)
labely = sio.loadmat('./dataset/2000/meshy/meshy1')['b'].reshape(2000,64,64)
data_input = get_data(bx, by)
label = get_data(labelx, labely)
data_input = torch.from_numpy(data_input).type(torch.FloatTensor).to(device)
net1 = UNetV2(in_channels=2, num_classes=2).to(device)
net2 = UNetV2(in_channels=2, num_classes=2).to(device)
net1.load_state_dict(torch.load("path/Mesh/best_mesh_phyaccnorm2000.pth"))
net2.load_state_dict(torch.load("path/Mesh/best_mesh_phyaccnorm2000mse.pth"))
mae_unet = 0
mae_fno = 0
mre_unet = 0
mre_fno = 0

for i,item in enumerate(data_input):
    print(i)
    F = item.unsqueeze(0)
    with torch.no_grad():
        fno_p = get_result_norm(net1,F)
        unet_p = get_result_norm(net2,F)
        # unet_p= get_result(net2(F),F)  
        # print(unet_p.shape)
    fno = fno_p[0].cpu().numpy()
    unet = unet_p[0].cpu().numpy()
    error_fno = np.abs(fno-label[i])
    error_unet = np.abs(unet-label[i])
    mae_unet = mae_unet + np.mean(error_unet)
    mae_fno = mae_fno + np.mean(error_fno)
    mre_unet = mre_unet + np.mean(error_unet/np.abs(label[i]+1e-10))
    mre_fno = mre_fno + np.mean(error_fno/np.abs(label[i]+1e-10))
    # error.append(np.mean(error_fno))
print(mae_unet/label.shape[0],mre_unet/label.shape[0])
print(mae_fno/label.shape[0],mre_fno/label.shape[0])
