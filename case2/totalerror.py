import numpy as np
import scipy.io as sio
import torch
from model.Unet.unet import UNetV2

def get_data(data1,data2):
    data = np.zeros((1,2,64,64))
    data[0][0] = data1
    data[0][1] = data2
    return data

def get_result_norm(net,data):
    data_input = torch.empty_like(data)
    x = 100
    y = 90
    data_input[0][0] = data[0][0] / x
    data_input[0][1] = data[0][1] / y
    data_output = net(data_input)
    data_output[0][0] = data_output[0][0] * x
    data_output[0][1] = data_output[0][1] * y
    result = data.clone()
    result[...,1:-1,1:-1] = data_output[...,1:-1,1:-1]
    return result

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
T = sio.loadmat('dataset/2000/T/Neuman/T1.mat')['b'].reshape(2000,64,64)
J = sio.loadmat('dataset/2000/J/J1')['b']
bx = sio.loadmat('dataset/2000/bound/boundx1')['b']
by = sio.loadmat('dataset/2000/bound/boundy1')['b']
meshx = sio.loadmat('dataset/2000/meshx/meshx1')['b']
meshy = sio.loadmat('dataset/2000/meshy/meshy1')['b']
net = UNetV2(in_channels=2, num_classes=2).to(device)
net.load_state_dict(torch.load("path/Mesh/best_mesh_phyaccnorm2000.pth"))

error = np.array([])
erroru = np.array([])
errormesh = np.array([])
error_tx = np.array([])
error_ty = np.array([])
error_mx = np.array([])
error_my = np.array([])
for i,item in enumerate(T):
    F = get_data(bx[i],by[i])
    u = get_data(meshx[i],meshy[i])
    F = torch.from_numpy(F).type(torch.FloatTensor).to(device)
    item = item[1:-1,1:-1]
    with torch.no_grad():
        y1 = get_result_norm(net,F)
    dx = np.abs(y1[0][0].cpu().numpy()-u[0][0])[1:-1,1:-1]
    dy = np.abs(y1[0][1].cpu().numpy()-u[0][1])[1:-1,1:-1]
    Txi = np.gradient(item)[0]
    yxi = np.gradient(meshy[i][1:-1,1:-1])[0]
    xxi = np.gradient(meshx[i][1:-1,1:-1])[0]
    Teta = np.gradient(item)[1]
    yeta = np.gradient(meshy[i][1:-1,1:-1])[1]
    xeta = np.gradient(meshx[i][1:-1,1:-1])[1]
    Tx = (1 / J[i][1:-1,1:-1]) * (Txi * yeta - Teta * yxi)
    Ty = (1 / J[i][1:-1,1:-1]) * (Teta * xxi - Txi * xeta)
    u = np.max(np.sqrt((Tx)**2+(Ty)**2))
    mesh = np.max(np.sqrt((dx)**2+(dy)**2))
    erroru = np.append(erroru,u)
    errormesh= np.append(errormesh,mesh)
    # error_tx = np.append(error_tx,np.max(Tx))
    # error_ty = np.append(error_ty,np.max(Ty))
    # error_mx = np.append(error_mx,np.max(dx))
    # error_my = np.append(error_my,np.max(dy))
    # print(np.max(Tx*dx))
print(np.mean(errormesh))
print(np.mean(erroru))