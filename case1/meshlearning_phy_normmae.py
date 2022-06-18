from os import error
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from torch.utils import data
import torch.utils.data as tud
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model.Unet.unet import UNetV2,UNet
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tr_writer = SummaryWriter(log_dir="runs/mesh_phy_norm/train")
te_writer = SummaryWriter(log_dir="runs/mesh_phy_norm/test")
tags = ['train_Loss','learning_rate']
tags1 = ['test_Loss']

def get_data(data1,data2):
    ny = data1[0].shape[0]
    nx = data1[0].shape[1]
    data = np.zeros((2000,2,ny,nx))
    for i in range(len(data)):
        x = np.max(data1[i])
        # y = np.max(data2[i])
        data[i][0] = data1[i] / x
        data[i][1] = data2[i] / x
    return data

# def norm(data):
#     temp = np.empty_like(data)
#     for i in range(len(data)):
#         x = np.max(data[i][0])
#         y = np.max(data[i][1])
#         temp[i][0] = data[i][0] / x
#         temp[i][1] = data[i][1] / y
#     return temp

def pde_loss(output,data,h=1):
    sumx = 0
    sumy = 0
    output[...,0,:] = data[...,0,:].clone()
    output[...,-1,:] = data[...,-1,:].clone()
    output[...,1:-1,0] = data[...,1:-1,0].clone()
    output[...,1:-1,-1] = data[...,1:-1,-1].clone()
    for i,item in enumerate(output):
        x = item[0] * torch.max(item[0])
        y = item[1] * torch.max(item[1])
        A = ((x[1:-1,2:]-x[1:-1,0:-2])/2/h)**2+\
            ((y[1:-1,2:]-y[1:-1,0:-2])/2/h)**2
        B = (x[2:,1:-1]-x[0:-2,1:-1])/2/h*\
            (x[1:-1,2:]-x[1:-1,0:-2])/2/h+\
            (y[2:,1:-1]-y[0:-2,1:-1])/2/h*\
            (y[1:-1,2:]-y[1:-1,0:-2])/2/h
        C = ((x[2:,1:-1]-x[0:-2,1:-1])/2/h)**2+\
            ((y[2:,1:-1]-y[0:-2,1:-1])/2/h)**2
        X = A*(x[2:,1:-1]+x[0:-2,1:-1]-2*x[1:-1,1:-1])/(h**2)+C*(x[1:-1,2:]+x[1:-1,0:-2]-2*x[1:-1,1:-1])/(h**2)-\
            B/2*(x[2:,2:]+x[0:-2,0:-2]-x[2:,0:-2]-x[0:-2,2:])/(h**2)
        Y = A*(y[2:,1:-1]+y[0:-2,1:-1]-2*y[1:-1,1:-1])/(h**2)+C*(y[1:-1,2:]+y[1:-1,0:-2]-2*y[1:-1,1:-1])/(h**2)-\
            B/2*(y[2:,2:]+y[0:-2,0:-2]-y[2:,0:-2]-y[0:-2,2:])/(h**2)
        sumx = sumx + torch.mean(torch.abs(X))
        sumy = sumy + torch.mean(torch.abs(Y))
    # sumx = sumx / output.shape[0]
    # sumy = sumy / output.shape[0]
    return sumx + sumy

def BRE(output,label):
    error = torch.abs(output - label)
    relative = error / torch.abs(label)
    brex = 0
    brey = 0
    for i,item in enumerate(relative):
        brex = brex + torch.mean(item[0])
        brey = brey + torch.mean(item[1])
    return brex, brey

def get_result(output,data):
    result = data.clone()
    result[...,1:-1,1:-1] = output[...,1:-1,1:-1]
    return result

class LayoutDataset(Dataset): 
    def __init__(self,train=True):
        data_input1 = sio.loadmat('./dataset/2000/bound/boundx1')['b']
        data_input2 = sio.loadmat('./dataset/2000/bound/boundy1')['b']
        data_output1 = sio.loadmat('./dataset/2000/mesh/meshx1')['b']
        data_output2 = sio.loadmat('./dataset/2000/mesh/meshy1')['b']
        data_input = get_data(data_input1, data_input2)
        data_output = get_data(data_output1,data_output2)
        # permutation = np.random.permutation(data_input.shape[0]) 
        # data_input = data_input[permutation]
        # data_output = data_output[permutation]
        if train:
            self.F = data_input[:1600]
            self.u = data_output[:1600]
        else:
            self.F = data_input[1600:]
            self.u = data_output[1600:]
    def __len__(self):
        return self.F.shape[0]
 
    def __getitem__(self, idx):
        f, o = self.F[idx], self.u[idx]         
        return f, o

def train(model,train_loader,optimizer,epoch):
    model.train()
    total_loss = 0.
    total_loss1 = 0.
    brex = 0
    brey = 0
    for idx, (data, label) in enumerate(train_loader):
        data, label = data.type(torch.FloatTensor).to(device), label.type(torch.FloatTensor).to(device)
        optimizer.zero_grad()
        output = model(data)
        result = get_result(output,data)
        Brex,Brey = BRE(result,label)
        loss1 = loss_function(result,label)
        loss = pde_loss(output,data)
        loss.backward()
        optimizer.step()
        brex += Brex.item()
        brey += Brey.item()
        total_loss += loss.item() * data.size(0)
        total_loss1 += loss1.item() * data.size(0)
    epoch_loss = total_loss / len(train_loader.dataset)
    total_loss1 = total_loss1 / len(train_loader.dataset)
    brex_mean = brex / len(train_loader.dataset)
    brey_mean = brey / len(train_loader.dataset)
    tr_writer.add_scalar(tags[0], epoch_loss, epoch)
    tr_writer.add_scalar(tags[1], optimizer.param_groups[0]['lr'], epoch)
    print("Train Epoch: {}, train_loss: {}, train_mae: {}, learning_rate: {}, brex_mean: {}, brey_mean: {}".format(epoch,epoch_loss,total_loss1,optimizer.param_groups[0]['lr'],brex_mean,brey_mean))

def test(model,test_loader,epoch):
    model.eval() # 不改变其权值
    total_loss = 0.
    total_loss1 = 0.
    brex = 0
    brey = 0
    with torch.no_grad():
        for idx, (data,label) in enumerate(test_loader):
            data, label = data.type(torch.FloatTensor).to(device), label.type(torch.FloatTensor).to(device)
            output = model(data)
            result = get_result(output,data)
            Brex,Brey = BRE(result,label)
            loss1 = loss_function(result,label)
            loss = pde_loss(output,data)
            brex += Brex.item()
            brey += Brey.item()
            total_loss += loss.item() * data.size(0)
            total_loss1 += loss1.item() * data.size(0)
    total_loss /= len(test_loader.dataset)
    total_loss1 /= len(test_loader.dataset)
    brex_mean = brex / len(test_loader.dataset)
    brey_mean = brey / len(test_loader.dataset)
    te_writer.add_scalar(tags1[0], total_loss, epoch)
    print("Test Epoch: {}, test_loss: {}, test_mae: {}, brex_mean: {}, brey_mean: {}".format(epoch,total_loss,total_loss1,brex_mean,brey_mean))
    return total_loss

batch_size = 16
train_data = LayoutDataset(train=True)
test_data = LayoutDataset(train=False)
train_loader = tud.DataLoader(train_data,batch_size = batch_size,shuffle=True) 
test_loader = tud.DataLoader(test_data,batch_size = batch_size) 

loss_function = nn.L1Loss()
net = UNetV2(in_channels=2, num_classes=2).to(device)
#optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, verbose=False)

num_epochs = 1000
best_loss = 900
for epoch in range(num_epochs):
    train(net,train_loader,optimizer,epoch)
    loss = test(net,test_loader,epoch)
    scheduler.step(loss)
    if loss < best_loss:
        best_loss = loss
        torch.save(net.state_dict(),"path/DeepLearning/best_mesh_phy_normmae.pth")