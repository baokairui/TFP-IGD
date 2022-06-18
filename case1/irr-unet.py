import torch
import time
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torch.utils.data as tud
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from model.Unet.unet import UNetV2
time_begin = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

datasize = 100

tr_writer = SummaryWriter(log_dir=f"runs/unet/train-{datasize}")
te_writer = SummaryWriter(log_dir=f"runs/unet/test-{datasize}")
tags = ['train_Loss','learning_rate']
tags1 = ['test_Loss']

def get_data(data1,data2):
    ny = data1[0].shape[0]
    nx = data1[0].shape[1]
    data = np.zeros((data1.shape[0],2,ny,nx))
    for i in range(len(data)):
        data[i][0] = data1[i]
        data[i][1] = data2[i]
    return data

class LayoutDataset(Dataset): 
    def __init__(self,nx,ny,train=True):
        self.nx = nx
        self.ny = ny
        data_input1 = sio.loadmat(f'./dataset/{datasize}/bound/boundx1')['b']
        data_input2 = sio.loadmat(f'./dataset/{datasize}/bound/boundy1')['b']
        data_output = sio.loadmat(f'./dataset/{datasize}/T/Dirichlet/T1')['b']
        data_input = get_data(data_input1, data_input2)
        data_output = data_output.reshape(data_input1.shape[0],1,self.ny,self.nx)
        permutation = np.random.permutation(data_input.shape[0])
        data_input = data_input[permutation]
        data_output = data_output[permutation]
        if train:
            self.F = data_input[:int(data_input1.shape[0]*0.7)]
            self.u = data_output[:int(data_input1.shape[0]*0.7)]
        else:
            self.F = data_input[int(data_input1.shape[0]*0.7):]
            self.u = data_output[int(data_input1.shape[0]*0.7):]
    def __len__(self):
        return self.F.shape[0]
 
    def __getitem__(self, idx):
        f, o = self.F[idx], self.u[idx]
        return f, o

def train(model,train_loader,optimizer,epoch):
    model.train()
    total_loss = 0.
    MRE = 0
    for idx, (data, label) in enumerate(train_loader):
        data, label = data.type(torch.FloatTensor).to(device), label.type(torch.FloatTensor).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output.to(device),label)
        loss.backward()
        optimizer.step()
        MRE += torch.mean(torch.abs(output-label) / label)
        total_loss += loss.item() * data.size(0)
    epoch_loss = total_loss / len(train_loader.dataset)
    MMRE = MRE / len(train_loader.dataset)
    tr_writer.add_scalar(tags[0], epoch_loss, epoch)
    tr_writer.add_scalar(tags[1], optimizer.param_groups[0]['lr'], epoch)
    print("Train Epoch: {}, train_loss: {}, learning_rate: {}, MRE:{}".format(epoch,epoch_loss,optimizer.param_groups[0]['lr'],MMRE))

def test(model,test_loader,epoch):
    model.eval() # 不改变其权值
    total_loss = 0.
    MRE = 0
    with torch.no_grad():
        for idx, (data,label) in enumerate(test_loader):
            data, label = data.type(torch.FloatTensor).to(device), label.type(torch.FloatTensor).to(device)
            output = model(data) # batch_size * 10
            loss = loss_function(output.to(device),label)
            MRE += torch.mean(torch.abs(output-label) / label)
            total_loss += float(loss)*data.size(0)
    total_loss /= len(test_loader.dataset)
    MMRE = MRE / len(train_loader.dataset)
    te_writer.add_scalar(tags1[0], total_loss, epoch)
    print("Test Epoch: {}, test_loss: {}, MRE: {}".format(epoch,total_loss,MMRE))
    return total_loss

batch_size = 8
train_data = LayoutDataset(128,32,train=True)
test_data = LayoutDataset(128,32,train=False)
train_loader = tud.DataLoader(train_data,batch_size = batch_size,shuffle=True) 
test_loader = tud.DataLoader(test_data,batch_size = batch_size)

loss_function = nn.L1Loss()
net = UNetV2(in_channels=2, num_classes=1).to(device)
#optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=False)

num_epochs = 200
best_loss = 0.9
for epoch in range(num_epochs):
    train(net,train_loader,optimizer,epoch)
    loss = test(net,test_loader,epoch)
    scheduler.step(loss)
    if loss < best_loss:
        best_loss = loss
        torch.save(net.state_dict(),f"path/Thermal/irr-unet-{datasize}.pth")

time_end = time.time()
time = time_end - time_begin
print(time)
np.savetxt(f'result/time/Timeunet{datasize}.txt',np.zeros([2,2])+time)