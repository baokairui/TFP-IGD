import os
import sys
from numpy.lib.arraypad import pad
model_dir = os.path.abspath('./model/FNO')
sys.path.append(model_dir)
import numpy as np
import matplotlib.pyplot as plt
import torch
from pyMesh import hcubeMesh, plotBC, visualize2D, plotMesh,setAxisLabel
from ML_mesh_parameter import get_bezier
from FD_solver import solver
from pod import get_new_pre,singlelevel
from model.Unet.unet import UNetV2
from model.FNO.fourier_2d import FNO2d
from matplotlib import rcParams

config = {
    "font.family":'serif',
    "font.size": 20,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)
plt.rc('font', family='Times New Roman')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class plot_result(object):
    def __init__(self,nx,ny,left_bound,right_bound,x_bound,y1,y2,clow='green',cup='blue',cright='red',cleft='orange',h=0.01,level=4):
        self.nx = nx
        self.ny = ny
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.x_bound = x_bound
        self.y1 = y1
        self.y2 = y2
        self.clow = clow
        self.cup = cup
        self.cright = cright
        self.cleft = cleft
        self.h = h
        self.level = level
    
    def get_bound(self):
        upx,upy = get_bezier(self.y1,self.y2,self.nx,self.x_bound,self.left_bound,self.right_bound)
        lowy = -upy
        lowx = upx
        lefty = np.linspace(-self.left_bound,self.left_bound,self.ny)
        righty = np.linspace(-self.right_bound,self.right_bound,self.ny)
        leftx = np.ones_like(lefty) * -self.x_bound
        rightx = np.ones_like(righty)* self.x_bound
        return upx,upy,lowx,lowy,leftx,lefty,rightx,righty

    def get_data(self,data1,data2):
        data = np.zeros((1,2,self.ny,self.nx))
        data[0][0] = data1
        data[0][1] = data2
        return data

    def plot_mesh(self):
        upx,upy,lowx,lowy,leftx,lefty,rightx,righty = plot_result.get_bound(self)
        myMesh = hcubeMesh(leftx,lefty,rightx,righty,lowx,lowy,upx,upy,self.h,True,True,saveDir='result/mesh.svg',tolMesh=1e-10,tolJoint=1)
        return myMesh

    def get_result(self, output, data):
        result = data.clone()
        result[...,1:-1,1:-1] = output[...,1:-1,1:-1]
        return result

    def get_result_norm(self,net,data):
        data_input = torch.empty_like(data)
        # x = torch.max(data[0][0])
        # y = torch.max(data[0][1])
        x = 100
        y = 50
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

    # def plot(self,ax,x,y,result,bound,label1,label2,path):
    #     fig=plt.figure()
    #     visualize2D(ax,x,y,result,'horizontal', bound)
    #     setAxisLabel(ax,label1)
    #     ax.set_title(label2)
    #     ax.set_aspect('equal')
    #     fig.tight_layout(pad=1)
    #     fig.savefig(path,bbox_inches='tight')
    def plot_bound(self,upx,upy,lowx,lowy,leftx,lefty,rightx,righty):
        x = np.zeros([self.ny,self.nx])
        y = np.zeros([self.ny,self.nx])
        plt.figure(figsize=(2.2,1))
        x[:,0] = leftx; y[:,0] = lefty
        x[:,-1] = rightx; y[:,-1] = righty
        x[0,:] = lowx; y[0,:] = lowy
        x[-1,:] = upx; y[-1,:] = upy
        plt.plot(x[:,0],y[:,0],color=self.cright)    # left BC
        plt.plot(x[:,-1],y[:,-1],color=self.cright) # right BC
        plt.plot(x[0,:],y[0,:],color=self.cright)    	# low BC
        plt.plot(x[-1,:],y[-1,:],color=self.cup)  	# up BC
        plt.axis('off')
        plt.savefig('result/bound.pdf',bbox_inches='tight')
        plt.figure()
        plt.imshow(x,cmap='coolwarm')
        plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal',aspect=50)
        plt.axis('off')  #去掉坐标轴
        plt.title('input_x')
        # plt.set_title('input_x')
        # plt.set_aspect('equal')
        plt.savefig('result/bound_x.pdf',bbox_inches='tight')
        plt.figure()
        plt.imshow(y,cmap='coolwarm')
        plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal',aspect=50)
        plt.axis('off')  #去掉坐标轴
        plt.title('input_y')
        # plt.set_title('error_y')
        # plt.set_aspect('equal')
        plt.savefig('result/bound_y.pdf',bbox_inches='tight')
        return x,y
    
    def plot_reg_result(self):
        fd = solver(self.nx,self.ny,self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2)
        result = fd.get_result()[0]
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        plt.imshow(result,cmap="coolwarm")
        ax.set_title('regular FD'+r'$T$')
        ax.set_aspect('equal')
        fig.tight_layout(pad=1)
        fig.savefig('result/FD_reg_result.pdf',bbox_inches='tight')

    def plot_fd_result(self):
        fd = solver(self.nx,self.ny,self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2)
        result,myMesh = fd.get_result()
        # ax = plt.subplot(1,1,1)
        # plot_result.plot(self,ax,myMesh.x,myMesh.y,result,[22,60],'p','irregular FD'+r'$T$','result/FD_result.pdf')
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        visualize2D(ax,myMesh.x,myMesh.y,result,'horizontal',[22,60])
        setAxisLabel(ax,'p')
        ax.set_title('irregular FD'+r'$T$')
        ax.set_aspect('equal')
        fig.tight_layout(pad=1)
        fig.savefig('result/FD_result.pdf',bbox_inches='tight')
        return result

    def plot_ml_result(self):
        myMesh = plot_result.plot_mesh(self)
        prediction = get_new_pre('path/POD','dataset/POD',[self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2],self.level)
        model_total = prediction.load_model()
        mean, base = prediction.get_meanbase()
        result = prediction.multilevel_result(model_total,mean,base)
        # print(result.shape)
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        visualize2D(ax,myMesh.x,myMesh.y,result,'horizontal',[22,60])
        # setAxisLabel(ax,'p')
        ax.set_title('irregular ML '+r'$T$')
        ax.set_aspect('equal')
        fig.tight_layout(pad=1)
        fig.savefig('result/ML_result.pdf',bbox_inches='tight')
        return result

    def plot_fdml_error(self):
        fd = solver(self.nx,self.ny,self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2)
        tem,myMesh = fd.get_result()
        prediction = get_new_pre('path/POD/1000','dataset/1000/POD',[self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2],self.level)
        model_total = prediction.load_model()
        mean, base = prediction.get_meanbase()
        result = prediction.multilevel_result(model_total,mean,base)
        fig = plt.figure(figsize=(12,6))
        ax = plt.subplot(1,3,1)
        visualize2D(ax,myMesh.x,myMesh.y,tem,'horizontal',[20,60])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('FD',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax=plt.subplot(1,3,2)
        visualize2D(ax,myMesh.x,myMesh.y,result,'horizontal',[20,60])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('ML',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax=plt.subplot(1,3,3)
        visualize2D(ax,myMesh.x,myMesh.y,result.reshape(self.ny,self.nx)-tem,'horizontal',[-0.2,0.2])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('error',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        fig.tight_layout(pad=1)
        fig.savefig('result/error_mlfd.pdf',bbox_inches='tight')

    def plot_singleml_error(self):
        fd = solver(self.nx,self.ny,self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2)
        tem,myMesh = fd.get_result()
        prediction = get_new_pre('path/POD/1000','dataset/1000/POD',[self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2],self.level)
        model_total = prediction.load_model()
        mean, base = prediction.get_meanbase()
        result = prediction.multilevel_result(model_total,mean,base)
        single_level = singlelevel('./dataset/1000/T/Dirichlet/T1.mat','./dataset/para-1000_new',0.999,1000)
        single_mean, single_base = single_level.get_meanbase()
        result_single = single_level.get_newpre([self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2],single_mean, single_base).reshape(32,128)
        fig = plt.figure(figsize=(10,6))
        ax = plt.subplot(2,3,1)
        visualize2D(ax,myMesh.x,myMesh.y,tem,'horizontal',[20,60])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('FD',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,2)
        visualize2D(ax,myMesh.x,myMesh.y,result,'horizontal',[20,60])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('ML',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,3)
        visualize2D(ax,myMesh.x,myMesh.y,result.reshape(self.ny,self.nx)-tem,'horizontal',[-0.2,0.2])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('error',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,4)
        visualize2D(ax,myMesh.x,myMesh.y,tem,'horizontal',[20,60])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('FD',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,5)
        visualize2D(ax,myMesh.x,myMesh.y,result_single,'horizontal',[20,60])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('SL',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,6)
        visualize2D(ax,myMesh.x,myMesh.y,result_single.reshape(self.ny,self.nx)-tem,'horizontal',[-0.2,0.2])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('error',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        fig.tight_layout(pad=1)
        fig.savefig('result/error_singleml.pdf',bbox_inches='tight')
        # print(np.mean(np.abs(result.reshape(self.ny,self.nx)-tem)))

    def plot_podbase(self,level):
        prediction = get_new_pre('path/POD','dataset/POD',[self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2],self.level)
        base = prediction.get_meanbase()[1]
        fig = plt.figure()
        for i,item in enumerate(base):
            for j,b in enumerate(item):
                if i == level:
                    x = int(64 / (2**(3-i)))
                    y = int(256 / (2**(3-i)))
                    temp = b.reshape(x,y)
                    plt.subplot(1,item.shape[0],j+1)
                    plt.imshow(temp,cmap='coolwarm')
                    plt.xticks([])
                    plt.yticks([])
                    fig.savefig('result/pod_base'+str(level)+'.pdf')
    
    def plot_podcal(self):
        prediction = get_new_pre('path/POD','dataset/POD',[self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2],self.level)
        model_total = prediction.load_model()
        mean, base = prediction.get_meanbase()
        prediction.multilevel_resultshow(model_total,mean,base)
    
    def plot_multierror(self):
        fig = plt.figure()
        error1 = [0.045,0.023,0.016,0.015,0.014]
        error2 = [0.052,0.039,0.020,0.016,0.015]
        # error3 = [0.11,0.05,0.039,0.037,0.036]
        # error4 = [0.13,0.10,0.05,0.04,0.037]
        y_level1 = [0.01,0.02,0.03,0.04,0.05,0.06]
        # y_level2 = [0.01,0.03,0.06,0.09,0.12,0.15]
        x_level = [100,200,500,1000,1500]
        plt.plot(x_level,error1,"-ro",label='ML')
        plt.plot(x_level,error2,"-bo",label='SL')
        plt.xticks(x_level,fontproperties = 'Times New Roman', size = 14)
        plt.yticks(y_level1,fontproperties = 'Times New Roman', size = 14)
        plt.ylabel('MAE$(\it{K})$',fontdict={'family' : 'Times New Roman', 'size' : 16})
        plt.xlabel('Dataset Scale',fontdict={'family' : 'Times New Roman', 'size' : 16})
        plt.legend(prop={'family' : 'Times New Roman', 'size' : 14})
        # ax2 = ax1.twinx()
        # ax2.plot(x_level,error3,"-.ro",label='ML(MRE)')
        # ax2.plot(x_level,error4,"-.bo",label='SL(MRE)')
        # plt.ylabel('MRE$(\%)$',fontdict={'family' : 'Times New Roman', 'size' : 16})
        # ax2.legend(loc=2)
        # plt.yticks(y_level2,fontproperties = 'Times New Roman', size = 14)
        # plt.axes(yscale = "log")                # 在plot语句前加上该句话即可
        # plt.plot(x_level,error1,"-ro",label='Multi-level model(ours)')
        # plt.plot(x_level,error2,"-bo",label='U-net model')
        # plt.plot(x_level,error3,"-go",label='FNO model')
        # plt.xlabel('Dataset Scale',fontdict={'family' : 'Times New Roman', 'size' : 16})
        # plt.ylabel('MAE$(\it{K})$',fontdict={'family' : 'Times New Roman', 'size' : 16})
        # # plt.title(station, fontdict={'family' : 'Times New Roman', 'size'   : 16})
        # plt.xticks(x_level,fontproperties = 'Times New Roman', size = 14)
        # plt.yticks(y_level,fontproperties = 'Times New Roman', size = 14)
        # plt.legend(prop={'family' : 'Times New Roman', 'size' : 14})
        fig.tight_layout(pad=1)
        fig.savefig('result/MAE.pdf')

        # fig1 = plt.figure()
        # error1 = [0.038,0.032,0.031,0.029,0.030]
        # error2 = [0.164,0.093,0.062,0.036,0.033]
        # error3 = [0.162,0.090,0.059,0.034,0.033]
        # y_level = [0.03,0.06,0.09,0.12,0.15,0.18]
        # x_level = [500,1000,1500,2000,2500]
        # # plt.axes(yscale = "log")                # 在plot语句前加上该句话即可
        # plt.plot(x_level,error1,"-r*",label='Multi-level model(ours)')
        # plt.plot(x_level,error2,"-b*",label='U-net model')
        # plt.plot(x_level,error3,"-g*",label='FNO model')
        # plt.xlabel('Dataset Scale',fontdict={'family' : 'Times New Roman', 'size' : 16})
        # plt.ylabel('MRE$(\%)$',fontdict={'family' : 'Times New Roman', 'size' : 16})
        # # plt.title(station, fontdict={'family' : 'Times New Roman', 'size'   : 16})
        # plt.xticks(x_level,fontproperties = 'Times New Roman', size = 14)
        # plt.yticks(y_level,fontproperties = 'Times New Roman', size = 14)
        # plt.legend(prop={'family' : 'Times New Roman', 'size' : 14})
        # fig1.tight_layout(pad=1)
        # fig1.savefig('result/MRE.pdf')
    def plot_mlunetfno_error(self):
        fd = solver(self.nx,self.ny,self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2)
        tem,myMesh = fd.get_result()
        prediction = get_new_pre('path/POD/1000','dataset/1000/POD',[self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2],self.level)
        model_total = prediction.load_model()
        mean, base = prediction.get_meanbase()
        result = prediction.multilevel_result(model_total,mean,base)
        upx,upy,lowx,lowy,leftx,lefty,rightx,righty = plot_result.get_bound(self)
        x, y = plot_result.plot_bound(self,upx,upy,lowx,lowy,leftx,lefty,rightx,righty)
        F = plot_result.get_data(self,x,y)
        F = torch.from_numpy(F).type(torch.FloatTensor).to(device)
        net1 = UNetV2(in_channels=2, num_classes=1).to(device)
        net1.load_state_dict(torch.load("path/Thermal/irr-unet-1000.pth"))
        net2 = FNO2d(12, 12, 32, input_channels=2).to(device)
        net2.load_state_dict(torch.load("path/Thermal/irr-fno-1000.pth"))
        with torch.no_grad():
            y1 = net1(F)
            y2 = net2(F)
        fig = plt.figure(figsize=(10,8))
        ax = plt.subplot(3,3,1)
        visualize2D(ax,myMesh.x,myMesh.y,tem,'horizontal',[20,60])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('FD',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax = plt.subplot(3,3,2)
        visualize2D(ax,myMesh.x,myMesh.y,result,'horizontal',[20,60])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('ML',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax = plt.subplot(3,3,3)
        visualize2D(ax,myMesh.x,myMesh.y,result.reshape(self.ny,self.nx)-tem,'horizontal',[-0.2,0.2])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('error',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax=plt.subplot(3,3,4)
        visualize2D(ax,myMesh.x,myMesh.y,tem,'horizontal',[20,60])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('FD',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax=plt.subplot(3,3,5)
        visualize2D(ax,myMesh.x,myMesh.y,y1[0][0].cpu().numpy(),'horizontal',[20,60])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('U-net',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax=plt.subplot(3,3,6)
        visualize2D(ax,myMesh.x,myMesh.y,y1[0][0].cpu().numpy()-result,'horizontal',[-0.2,0.2])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('error',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax=plt.subplot(3,3,7)
        visualize2D(ax,myMesh.x,myMesh.y,tem,'horizontal',[20,60])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('FD',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax=plt.subplot(3,3,8)
        visualize2D(ax,myMesh.x,myMesh.y,y2[0][0].cpu().numpy(),'horizontal',[20,60])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('FNO',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax=plt.subplot(3,3,9)
        visualize2D(ax,myMesh.x,myMesh.y,y2[0][0].cpu().numpy()-result,'horizontal',[-0.2,0.2])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('error',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        fig.tight_layout(pad=1)
        fig.savefig('result/error_unetfnofd.pdf',bbox_inches='tight')
    def plot_mlunet_error(self):
        fd = solver(self.nx,self.ny,self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2)
        tem,myMesh = fd.get_result()
        prediction = get_new_pre('path/POD/1000','dataset/1000/POD',[self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2],self.level)
        model_total = prediction.load_model()
        mean, base = prediction.get_meanbase()
        result = prediction.multilevel_result(model_total,mean,base)
        upx,upy,lowx,lowy,leftx,lefty,rightx,righty = plot_result.get_bound(self)
        x, y = plot_result.plot_bound(self,upx,upy,lowx,lowy,leftx,lefty,rightx,righty)
        F = plot_result.get_data(self,x,y)
        F = torch.from_numpy(F).type(torch.FloatTensor).to(device)
        net = UNetV2(in_channels=2, num_classes=1).to(device)
        net.load_state_dict(torch.load("path/Thermal/irr-unet-1000.pth"))
        with torch.no_grad():
            y = net(F)
        fig = plt.figure(figsize=(10,6))
        ax = plt.subplot(2,3,1)
        visualize2D(ax,myMesh.x,myMesh.y,tem,'horizontal',[20,60])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('FD',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax = plt.subplot(2,3,2)
        visualize2D(ax,myMesh.x,myMesh.y,result,'horizontal',[20,60])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('ML',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax = plt.subplot(2,3,3)
        visualize2D(ax,myMesh.x,myMesh.y,result.reshape(self.ny,self.nx)-tem,'horizontal',[-0.2,0.2])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('error',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,4)
        visualize2D(ax,myMesh.x,myMesh.y,tem,'horizontal',[20,60])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('FD',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,5)
        visualize2D(ax,myMesh.x,myMesh.y,y[0][0].cpu().numpy(),'horizontal',[20,60])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('U-net',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,6)
        visualize2D(ax,myMesh.x,myMesh.y,y[0][0].cpu().numpy()-result,'horizontal',[-0.2,0.2])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('error',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        fig.tight_layout(pad=1)
        fig.savefig('result/error_unetfd.pdf',bbox_inches='tight')
        

    def plot_mlfno_error(self):
        fd = solver(self.nx,self.ny,self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2)
        tem,myMesh = fd.get_result()
        prediction = get_new_pre('path/POD/1000','dataset/1000/POD',[self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2],self.level)
        model_total = prediction.load_model()
        mean, base = prediction.get_meanbase()
        result = prediction.multilevel_result(model_total,mean,base)
        upx,upy,lowx,lowy,leftx,lefty,rightx,righty = plot_result.get_bound(self)
        x, y = plot_result.plot_bound(self,upx,upy,lowx,lowy,leftx,lefty,rightx,righty)
        F = plot_result.get_data(self,x,y)
        F = torch.from_numpy(F).type(torch.FloatTensor).to(device)
        net = FNO2d(12, 12, 32, input_channels=2).to(device)
        net.load_state_dict(torch.load("path/Thermal/irr-fno-1000.pth"))
        with torch.no_grad():
            y = net(F)
        fig = plt.figure(figsize=(10,6))
        # plt.imshow(x,cmap='coolwarm')
        # plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal',aspect=50)
        # plt.axis('off')  #去掉坐标轴
        # plt.title('input_x')
        ax = plt.subplot(2,3,1)
        visualize2D(ax,myMesh.x,myMesh.y,tem,'horizontal',[20,60])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('FD',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax = plt.subplot(2,3,2)
        visualize2D(ax,myMesh.x,myMesh.y,result,'horizontal',[20,60])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('ML',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax = plt.subplot(2,3,3)
        visualize2D(ax,myMesh.x,myMesh.y,result.reshape(self.ny,self.nx)-tem,'horizontal',[-0.2,0.2])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('error',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,4)
        visualize2D(ax,myMesh.x,myMesh.y,tem,'horizontal',[20,60])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('FD',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,5)
        visualize2D(ax,myMesh.x,myMesh.y,y[0][0].cpu().numpy(),'horizontal',[20,60])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('FNO',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,6)
        visualize2D(ax,myMesh.x,myMesh.y,y[0][0].cpu().numpy()-result,'horizontal',[-0.2,0.2])
        plt.tick_params(labelsize=12, rotation=0)
        # setAxisLabel(ax,'p')
        ax.set_title('error',fontdict={'family' : 'Times New Roman','size': 15})
        ax.set_aspect('equal')
        fig.tight_layout(pad=1)
        fig.savefig('result/error_fnofd.pdf',bbox_inches='tight')
    
    def plot_meshlearning(self):
        upx,upy,lowx,lowy,leftx,lefty,rightx,righty = plot_result.get_bound(self)
        x, y = plot_result.plot_bound(self,upx,upy,lowx,lowy,leftx,lefty,rightx,righty)
        myMesh = plot.plot_mesh()
        F = plot_result.get_data(self,x,y)
        u = plot_result.get_data(self,myMesh.x,myMesh.y)
        F = torch.from_numpy(F).type(torch.FloatTensor).to(device)
        net1 = UNetV2(in_channels=2, num_classes=2).to(device)
        net2 = UNetV2(in_channels=2, num_classes=2).to(device)
        net1.load_state_dict(torch.load("path/DeepLearning/best_mesh_phyaccnorm1000.pth"))
        net2.load_state_dict(torch.load("path/DeepLearning/best_mesh_data1000.pth"))
        with torch.no_grad():
            y1 = plot_result.get_result_norm(self,net1,F)
            y2 = plot_result.get_result(self,net2(F),F)
        fig1 = plt.figure(figsize=(8,3))
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.05,hspace=0.05)
        ax1 = plt.subplot(2, 4, 1)
        plt.imshow(x,cmap='coolwarm')
        cbar = plt.colorbar(fraction=0.08, pad=0.05,orientation='horizontal')
        cbar.ax.tick_params(labelsize=10)
        plt.axis('off')  #去掉坐标轴
        ax1.set_title('Input '+r'$\it{x}$',fontdict={'family' : 'Times New Roman','size': 12})
        ax1.set_aspect('equal')
        plt.tick_params(labelsize=10, rotation=0)
        ax1 = plt.subplot(2, 4, 2)
        plt.imshow(u[0][0],cmap='coolwarm')
        cbar = plt.colorbar(fraction=0.08, pad=0.05,orientation='horizontal')
        cbar.ax.tick_params(labelsize=10)
        plt.axis('off')  #去掉坐标轴
        ax1.set_title('Ref '+r'$\it{x}$',fontdict={'family' : 'Times New Roman','size': 12})
        ax1.set_aspect('equal')
        plt.tick_params(labelsize=10, rotation=0)
        ax1 = plt.subplot(2, 4, 3)  
        plt.imshow(y1[0][0].cpu().numpy(),cmap='coolwarm')
        cbar = plt.colorbar(fraction=0.08, pad=0.05,orientation='horizontal')
        cbar.ax.tick_params(labelsize=10)
        plt.axis('off')  #去掉坐标轴
        ax1.set_title('Pre '+r'$\it{x}$',fontdict={'family' : 'Times New Roman','size': 12})
        ax1.set_aspect('equal')
        plt.tick_params(labelsize=10, rotation=0)
        ax1 = plt.subplot(2, 4, 4)  
        # plt.figure(dpi=500,figsize=(24,8))                                            #绘制一张空白图
        plt.imshow(np.abs(y1[0][0].cpu().numpy()-u[0][0]),cmap='coolwarm',vmin=-0.1, vmax=0.1)
        cb1 = plt.colorbar(fraction=0.08, pad=0.05,orientation='horizontal')
        cb1.ax.tick_params(labelsize=10)
        cb1.set_ticks([-0.1, 0, 0.1])     
        cb1.update_ticks()
        plt.axis('off')  #去掉坐标轴
        ax1.set_title('error ' +r'$\it{x}$',fontdict={'family' : 'Times New Roman','size': 12})
        ax1.set_aspect('equal')
        plt.tick_params(labelsize=10, rotation=0)
        ax1 = plt.subplot(2, 4, 5)
        plt.imshow(y,cmap='coolwarm')
        cbar = plt.colorbar(fraction=0.08, pad=0.05,orientation='horizontal')
        cbar.ax.tick_params(labelsize=10)
        plt.axis('off')  #去掉坐标轴
        ax1.set_title('Input '+r'$\it{y}$',fontdict={'family' : 'Times New Roman','size': 12})
        ax1.set_aspect('equal')
        plt.tick_params(labelsize=10, rotation=0)
        ax1 = plt.subplot(2, 4, 6)
        plt.imshow(u[0][1],cmap='coolwarm')
        cbar = plt.colorbar(fraction=0.08, pad=0.05,orientation='horizontal')
        cbar.ax.tick_params(labelsize=10)
        plt.axis('off')  #去掉坐标轴
        ax1.set_title('Ref '+r'$\it{y}$',fontdict={'family' : 'Times New Roman','size': 12})
        ax1.set_aspect('equal')
        plt.tick_params(labelsize=10, rotation=0)
        ax1 = plt.subplot(2, 4, 7)  
        plt.imshow(y1[0][1].cpu().numpy(),cmap='coolwarm')
        cbar = plt.colorbar(fraction=0.08, pad=0.05,orientation='horizontal')
        cbar.ax.tick_params(labelsize=10)
        plt.axis('off')  #去掉坐标轴
        ax1.set_title('Pre '+r'$\it{y}$',fontdict={'family' : 'Times New Roman','size': 12})
        ax1.set_aspect('equal')
        plt.tick_params(labelsize=10, rotation=0)
        ax1 = plt.subplot(2, 4, 8)  
        # plt.figure(dpi=500,figsize=(24,8))                                            #绘制一张空白图
        plt.imshow(np.abs(y1[0][1].cpu().numpy()-u[0][1]),cmap='coolwarm',vmin=-0.1, vmax=0.1)
        cb1 = plt.colorbar(fraction=0.08, pad=0.05,orientation='horizontal')
        cb1.ax.tick_params(labelsize=10)
        cb1.set_ticks([-0.1, 0, 0.1]) 
        cb1.update_ticks()
        plt.axis('off')  #去掉坐标轴
        ax1.set_title('error '+r'$\it{y}$',fontdict={'family' : 'Times New Roman','size': 12})
        ax1.set_aspect('equal')
        plt.tick_params(labelsize=10, rotation=0)
        fig1.tight_layout(pad=1)
        fig1.savefig('result/mesh_reg_err.pdf',bbox_inches='tight')

        fig2 = plt.figure()
        # plt.rc('font',family='Times New Roman', size=12)
        ax2=plt.subplot(1,2,1)
        plotMesh(ax2,u[0][0],u[0][1])
        # ax2.axis('off')
        ax2.set_title('Mesh Reference', fontdict={'family' : 'Times New Roman','size': 14})
        ax2.set_aspect('equal')
        plt.tick_params(labelsize=8, rotation=0)
        x1_label = ax2.get_xticklabels() 
        [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
        y1_label = ax2.get_yticklabels() 
        [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
        ax2.tick_params(axis='y',labelsize=8) 
        ax2.tick_params(axis='x',labelsize=8) 
        ax2=plt.subplot(1,2,2)
        plotMesh(ax2,y1[0][0].cpu(),y1[0][1].cpu())
        # ax2.axis('off')
        ax2.set_title('Mesh '+r'$\mathrm{Unet_{phy}}$', fontdict={'family' : 'Times New Roman','size': 14})
        ax2.set_aspect('equal')
        plt.tick_params(labelsize=8, rotation=0)
        x1_label = ax2.get_xticklabels() 
        [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
        y1_label = ax2.get_yticklabels() 
        [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
        ax2.tick_params(axis='y',labelsize=8) 
        ax2.tick_params(axis='x',labelsize=8) 
        # ax2=plt.subplot(1,3,3)
        # plotMesh(ax2,y2[0][0].cpu(),y2[0][1].cpu())
        # # ax2.axis('off')
        # ax2.set_title('Mesh '+r'$\mathrm{Unet_{data}}$', fontdict={'family' : 'Times New Roman','size': 14})
        # ax2.set_aspect('equal')
        # plt.tick_params(labelsize=8, rotation=0)
        # x1_label = ax2.get_xticklabels() 
        # [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
        # y1_label = ax2.get_yticklabels() 
        # [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
        # ax2.tick_params(axis='y',labelsize=8) 
        # ax2.tick_params(axis='x',labelsize=8) 
        
        # fig2.tight_layout(pad=1)
        # fig2.savefig('result/mesh_irr_err.svg',bbox_inches='tight')
        fig2.savefig('result/mesh_irr_err.pdf',bbox_inches='tight')

        # fig2 = plt.figure(figsize=(12,6))
        # # plt.rc('font',family='Times New Roman', size=12)
        # ax2=plt.subplot(2,3,1)
        # plotMesh(ax2,u[0][0],u[0][1])
        # # ax2.axis('off')
        # ax2.set_title('Mesh Reference', fontdict={'family' : 'Times New Roman','size': 14})
        # ax2.set_aspect('equal')
        # plt.tick_params(labelsize=8, rotation=0)
        # x1_label = ax2.get_xticklabels() 
        # [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
        # y1_label = ax2.get_yticklabels() 
        # [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
        # ax2.tick_params(axis='y',labelsize=8) 
        # ax2.tick_params(axis='x',labelsize=8) 
        # ax2=plt.subplot(2,3,2)
        # plotMesh(ax2,y1[0][0].cpu(),y1[0][1].cpu())
        # # ax2.axis('off')
        # ax2.set_title('Mesh '+r'$\mathrm{Unet_{phy}}$', fontdict={'family' : 'Times New Roman','size': 14})
        # ax2.set_aspect('equal')
        # plt.tick_params(labelsize=8, rotation=0)
        # x1_label = ax2.get_xticklabels() 
        # [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
        # y1_label = ax2.get_yticklabels() 
        # [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
        # ax2.tick_params(axis='y',labelsize=8) 
        # ax2.tick_params(axis='x',labelsize=8) 
        # ax2=plt.subplot(2,3,3)
        # plotMesh(ax2,y2[0][0].cpu(),y2[0][1].cpu())
        # # ax2.axis('off')
        # ax2.set_title('Mesh '+r'$\mathrm{Unet_{data}}$', fontdict={'family' : 'Times New Roman','size': 14})
        # ax2.set_aspect('equal')
        # plt.tick_params(labelsize=8, rotation=0)
        # x1_label = ax2.get_xticklabels() 
        # [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
        # y1_label = ax2.get_yticklabels() 
        # [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
        # ax2.tick_params(axis='y',labelsize=8) 
        # ax2.tick_params(axis='x',labelsize=8) 
        # ax2=plt.subplot(2,3,4)
        # plotMesh(ax2,u[0][0][:,:32],u[0][1][:,:32])
        # ax2.axis('off')

        # ax2=plt.subplot(2,3,5)
        # plotMesh(ax2,y1[0][0][:,:32].cpu(),y1[0][1][:,:32].cpu())
        # ax2.axis('off')

        # ax2=plt.subplot(2,3,6)
        # plotMesh(ax2,y2[0][0][:,:32].cpu(),y2[0][1][:,:32].cpu())
        # ax2.axis('off')

        # fig2.tight_layout(pad=1)
        # fig2.savefig('result/mesh_irr_err.svg',bbox_inches='tight')
        # fig2.savefig('result/mesh_irr_err.pdf',bbox_inches='tight')

if __name__ == '__main__':
    # plot = plot_result(128,32,16,75,140,0,45,level=3)
    # plot = plot_result(128,32,16,50,130,30,20,level=3)
    plot = plot_result(128,32,10,50,100,25,25,level=3)
    # plot = plot_result(128,32,15.12,59.33,129.38,17.93,28.03,level=3) #ml mesh
    # plot = plot_result(128,32,10.76,  56.31, 133.61,   5.89,  21.7,level=3) #SL
    # plot = plot_result(128,32,10.79,  30.48, 138.98,   9.29,  35.56,level=3) #ml
    # plot = plot_result(256,64,20,50,150,-10,50,level=4)

    # plot.plot_mesh()

    # upx,upy,lowx,lowy,leftx,lefty,rightx,righty = plot.get_bound()
    # plot.plot_bound(upx,upy,lowx,lowy,leftx,lefty,rightx,righty)

    # plot.plot_reg_result()

    # plot.plot_fd_result()

    # plot.plot_ml_result()

    # plot.plot_fdml_error()
    # plot.plot_singleml_error()

    # plot.plot_podbase(0)

    # plot.plot_podcal()

    # plot.plot_multierror()

    # plot.plot_mlunet_error()

    # plot.plot_mlfno_error()

    plot.plot_meshlearning()

    # plot.plot_mlunetfno_error()