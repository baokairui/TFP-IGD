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
        x = torch.max(data[0][0])
        y = torch.max(data[0][1])
        # print(x,y)
        data_input[0][0] = data[0][0] / x
        data_input[0][1] = data[0][1] / x
        # print(data_input[0][1])
        # net.load_state_dict(torch.load("path/DeepLearning/best_mesh.pth"))
        data_output = net(data_input)
        # print(data_output[0][1])
        # output = torch.empty_like(data_output)
        data_output[0][0] = data_output[0][0] * x
        data_output[0][1] = data_output[0][1] * x
        return data_output

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
        prediction = get_new_pre('path/POD/2000','dataset/2000/POD',[self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2],self.level)
        model_total = prediction.load_model()
        mean, base = prediction.get_meanbase()
        result = prediction.multilevel_result(model_total,mean,base)
        fig = plt.figure()
        ax = plt.subplot(1,3,1)
        visualize2D(ax,myMesh.x,myMesh.y,tem,'horizontal',[20,60])
        # setAxisLabel(ax,'p')
        ax.set_title('FD '+r'$T$')
        ax.set_aspect('equal')
        ax=plt.subplot(1,3,2)
        visualize2D(ax,myMesh.x,myMesh.y,result,'horizontal',[20,60])
        # setAxisLabel(ax,'p')
        ax.set_title('ML '+r'$T$')
        ax.set_aspect('equal')
        ax=plt.subplot(1,3,3)
        visualize2D(ax,myMesh.x,myMesh.y,result.reshape(self.ny,self.nx)-tem,'horizontal',[-1,1])
        # setAxisLabel(ax,'p')
        ax.set_title('error ML '+r'$T$')
        ax.set_aspect('equal')
        fig.tight_layout(pad=1)
        fig.savefig('result/error_mlfd.pdf',bbox_inches='tight')

    def plot_singleml_error(self):
        fd = solver(self.nx,self.ny,self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2)
        tem,myMesh = fd.get_result()
        prediction = get_new_pre('path/POD/2000','dataset/2000/POD',[self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2],self.level)
        model_total = prediction.load_model()
        mean, base = prediction.get_meanbase()
        result = prediction.multilevel_result(model_total,mean,base)
        single_level = singlelevel('./dataset/2000/T/T1.mat','./dataset/para-2000_new',0.999)
        single_mean, single_base = single_level.get_meanbase()
        result_single = single_level.get_newpre([self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2],single_mean, single_base).reshape(64,256)
        fig = plt.figure()
        ax = plt.subplot(2,3,1)
        visualize2D(ax,myMesh.x,myMesh.y,tem,'horizontal',[20,60])
        # setAxisLabel(ax,'p')
        ax.set_title('FD '+r'$T$')
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,2)
        visualize2D(ax,myMesh.x,myMesh.y,result,'horizontal',[20,60])
        # setAxisLabel(ax,'p')
        ax.set_title('ML '+r'$T$')
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,3)
        visualize2D(ax,myMesh.x,myMesh.y,result.reshape(self.ny,self.nx)-tem,'horizontal',[-0.1,0.1])
        # setAxisLabel(ax,'p')
        ax.set_title('error ML')
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,4)
        visualize2D(ax,myMesh.x,myMesh.y,tem,'horizontal',[20,60])
        # setAxisLabel(ax,'p')
        ax.set_title('FD '+r'$T$')
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,5)
        visualize2D(ax,myMesh.x,myMesh.y,result_single,'horizontal',[20,60])
        # setAxisLabel(ax,'p')
        ax.set_title('SL '+r'$T$')
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,6)
        visualize2D(ax,myMesh.x,myMesh.y,result_single.reshape(self.ny,self.nx)-tem,'horizontal',[-0.1,0.1])
        # setAxisLabel(ax,'p')
        ax.set_title('error SL')
        ax.set_aspect('equal')
        fig.tight_layout(pad=1)
        fig.savefig('result/error_singleml.pdf',bbox_inches='tight')
        print(np.mean(np.abs(result.reshape(self.ny,self.nx)-tem)))

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
        error1 = [0.133,0.043,0.0167]
        error2 = [0.135,0.045,0.0188]
        y_level = [0.1,0.01]
        x_level = [2,3,4]
        plt.axes(yscale = "log")                # 在plot语句前加上该句话即可
        plt.plot(x_level,error1,"-kh",label='Multi-level net(ours)')
        plt.plot(x_level,error2,"bh-.",label='Single net')
        plt.xlabel('Level',fontdict={'family' : 'Times New Roman', 'size' : 16})
        plt.ylabel('MAE',fontdict={'family' : 'Times New Roman', 'size' : 16})
        # plt.title(station, fontdict={'family' : 'Times New Roman', 'size'   : 16})
        plt.xticks(x_level,fontproperties = 'Times New Roman', size = 14)
        plt.yticks(y_level,fontproperties = 'Times New Roman', size = 14)
        plt.legend(prop={'family' : 'Times New Roman', 'size' : 16})
        fig.savefig('result/multi_pod_error.pdf')
    
    def plot_mlunet_error(self):
        fd = solver(self.nx,self.ny,self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2)
        tem,myMesh = fd.get_result()
        prediction = get_new_pre('path/POD/2000','dataset/2000/POD',[self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2],self.level)
        model_total = prediction.load_model()
        mean, base = prediction.get_meanbase()
        result = prediction.multilevel_result(model_total,mean,base)
        upx,upy,lowx,lowy,leftx,lefty,rightx,righty = plot_result.get_bound(self)
        x, y = plot_result.plot_bound(self,upx,upy,lowx,lowy,leftx,lefty,rightx,righty)
        F = plot_result.get_data(self,x,y)
        F = torch.from_numpy(F).type(torch.FloatTensor).to(device)
        net = UNetV2(in_channels=2, num_classes=1).to(device)
        net.load_state_dict(torch.load("path/DeepLearning/irr-unet-2000.pth"))
        with torch.no_grad():
            y = net(F)
        fig = plt.figure()
        ax = plt.subplot(2,3,1)
        visualize2D(ax,myMesh.x,myMesh.y,tem,'horizontal',[20,60])
        # setAxisLabel(ax,'p')
        ax.set_title('label '+r'$T$')
        ax.set_aspect('equal')
        ax = plt.subplot(2,3,2)
        visualize2D(ax,myMesh.x,myMesh.y,result,'horizontal',[20,60])
        # setAxisLabel(ax,'p')
        ax.set_title('ML '+r'$T$')
        ax.set_aspect('equal')
        ax = plt.subplot(2,3,3)
        visualize2D(ax,myMesh.x,myMesh.y,np.abs(result.reshape(self.ny,self.nx)-tem),'horizontal',[-0.1,0.1])
        # setAxisLabel(ax,'p')
        ax.set_title('error ML')
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,4)
        visualize2D(ax,myMesh.x,myMesh.y,tem,'horizontal',[20,60])
        # setAxisLabel(ax,'p')
        ax.set_title('label '+r'$T$')
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,5)
        visualize2D(ax,myMesh.x,myMesh.y,y[0][0].cpu().numpy(),'horizontal',[20,60])
        # setAxisLabel(ax,'p')
        ax.set_title('Unet '+r'$T$')
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,6)
        visualize2D(ax,myMesh.x,myMesh.y,np.abs(y[0][0].cpu().numpy()-result),'horizontal',[-0.1,0.1])
        # setAxisLabel(ax,'p')
        ax.set_title('error Unet')
        ax.set_aspect('equal')
        fig.tight_layout(pad=1)
        fig.savefig('result/error_unetfd.pdf',bbox_inches='tight')

    def plot_mlfno_error(self):
        fd = solver(self.nx,self.ny,self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2)
        tem,myMesh = fd.get_result()
        prediction = get_new_pre('path/POD/2000','dataset/2000/POD',[self.left_bound,self.right_bound,self.x_bound,self.y1,self.y2],self.level)
        model_total = prediction.load_model()
        mean, base = prediction.get_meanbase()
        result = prediction.multilevel_result(model_total,mean,base)
        upx,upy,lowx,lowy,leftx,lefty,rightx,righty = plot_result.get_bound(self)
        x, y = plot_result.plot_bound(self,upx,upy,lowx,lowy,leftx,lefty,rightx,righty)
        F = plot_result.get_data(self,x,y)
        F = torch.from_numpy(F).type(torch.FloatTensor).to(device)
        net = FNO2d(18, 18, 43, input_channels=2).to(device)
        net.load_state_dict(torch.load("path/DeepLearning/irr-fno-2000.pth"))
        with torch.no_grad():
            y = net(F)
        fig = plt.figure()
        # plt.imshow(x,cmap='coolwarm')
        # plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal',aspect=50)
        # plt.axis('off')  #去掉坐标轴
        # plt.title('input_x')
        ax = plt.subplot(2,3,1)
        visualize2D(ax,myMesh.x,myMesh.y,tem,'horizontal',[20,60])
        # setAxisLabel(ax,'p')
        ax.set_title('label '+r'$T$')
        ax.set_aspect('equal')
        ax = plt.subplot(2,3,2)
        visualize2D(ax,myMesh.x,myMesh.y,result,'horizontal',[20,60])
        # setAxisLabel(ax,'p')
        ax.set_title('ML'+r'$T$')
        ax.set_aspect('equal')
        ax = plt.subplot(2,3,3)
        visualize2D(ax,myMesh.x,myMesh.y,np.abs(result.reshape(self.ny,self.nx)-tem),'horizontal',[-0.1,0.1])
        # setAxisLabel(ax,'p')
        ax.set_title('error ML')
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,4)
        visualize2D(ax,myMesh.x,myMesh.y,tem,'horizontal',[20,60])
        # setAxisLabel(ax,'p')
        ax.set_title('label '+r'$T$')
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,5)
        visualize2D(ax,myMesh.x,myMesh.y,y[0][0].cpu().numpy(),'horizontal',[20,60])
        # setAxisLabel(ax,'p')
        ax.set_title('FNO'+r'$T$')
        ax.set_aspect('equal')
        ax=plt.subplot(2,3,6)
        visualize2D(ax,myMesh.x,myMesh.y,np.abs(y[0][0].cpu().numpy()-result),'horizontal',[-0.1,0.1])
        # setAxisLabel(ax,'p')
        ax.set_title('error FNO_pre')
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
        net1.load_state_dict(torch.load("path/DeepLearning/best_mesh_phy_new.pth"))
        net2.load_state_dict(torch.load("path/DeepLearning/best_mesh_data.pth"))
        with torch.no_grad():
            y1 = plot_result.get_result(self,net1(F),F)
            y2 = plot_result.get_result(self,net2(F),F)
        fig1 = plt.figure()
        ax1 = plt.subplot(2, 4, 1)
        plt.imshow(x,cmap='coolwarm')
        plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal')
        plt.axis('off')  #去掉坐标轴
        ax1.set_title('input_x')
        ax1.set_aspect('equal')
        ax1 = plt.subplot(2, 4, 2)
        plt.imshow(u[0][0],cmap='coolwarm')
        plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal')
        plt.axis('off')  #去掉坐标轴
        ax1.set_title('label_x')
        ax1.set_aspect('equal')
        ax1 = plt.subplot(2, 4, 3)  
        plt.imshow(y1[0][0].cpu().numpy(),cmap='coolwarm')
        plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal')
        plt.axis('off')  #去掉坐标轴
        ax1.set_title('pre_x')
        ax1.set_aspect('equal')
        ax1 = plt.subplot(2, 4, 4)  
        # plt.figure(dpi=500,figsize=(24,8))                                            #绘制一张空白图
        plt.imshow(np.abs(y1[0][0].cpu().numpy()-u[0][0]),cmap='coolwarm',vmin=-0.1, vmax=0.1)
        cb1 = plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal')
        cb1.set_ticks([-0.1, 0, 0.1])     
        cb1.update_ticks()
        plt.axis('off')  #去掉坐标轴
        ax1.set_title('error_x')
        ax1.set_aspect('equal')
        ax1 = plt.subplot(2, 4, 5)
        plt.imshow(y,cmap='coolwarm')
        plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal')
        plt.axis('off')  #去掉坐标轴
        ax1.set_title('input_y')
        ax1.set_aspect('equal')
        ax1 = plt.subplot(2, 4, 6)
        plt.imshow(u[0][1],cmap='coolwarm')
        plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal')
        plt.axis('off')  #去掉坐标轴
        ax1.set_title('label_y')
        ax1.set_aspect('equal')
        ax1 = plt.subplot(2, 4, 7)  
        plt.imshow(y1[0][1].cpu().numpy(),cmap='coolwarm')
        plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal')
        plt.axis('off')  #去掉坐标轴
        ax1.set_title('pre_y')
        ax1.set_aspect('equal')
        ax1 = plt.subplot(2, 4, 8)  
        # plt.figure(dpi=500,figsize=(24,8))                                            #绘制一张空白图
        plt.imshow(np.abs(y1[0][1].cpu().numpy()-u[0][1]),cmap='coolwarm',vmin=-0.1, vmax=0.1)
        cb1 = plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal')
        cb1.set_ticks([-0.1, 0, 0.1]) 
        cb1.update_ticks()
        plt.axis('off')  #去掉坐标轴
        ax1.set_title('error_y')
        ax1.set_aspect('equal')
        fig1.tight_layout(pad=1)
        fig1.savefig('result/mesh_reg_err.pdf')

        fig2 = plt.figure()
        plt.rc('font',family='Times New Roman', size=14)
        ax2=plt.subplot(1,3,1)
        plotMesh(ax2,u[0][0],u[0][1])
        # ax2.axis('off')
        ax2.set_title('Mesh Label')
        ax2.set_aspect('equal')
        x1_label = ax2.get_xticklabels() 
        [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
        y1_label = ax2.get_yticklabels() 
        [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
        ax2.tick_params(axis='y',labelsize=10) 
        ax2.tick_params(axis='x',labelsize=10) 
        ax2=plt.subplot(1,3,2)
        plotMesh(ax2,y1[0][0].cpu(),y1[0][1].cpu())
        # ax2.axis('off')
        ax2.set_title('Mesh '+r'$\mathrm{Unet_{phy}}$')
        ax2.set_aspect('equal')
        x1_label = ax2.get_xticklabels() 
        [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
        y1_label = ax2.get_yticklabels() 
        [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
        ax2.tick_params(axis='y',labelsize=10) 
        ax2.tick_params(axis='x',labelsize=10) 
        ax2=plt.subplot(1,3,3)
        plotMesh(ax2,y2[0][0].cpu(),y2[0][1].cpu())
        # ax2.axis('off')
        ax2.set_title('Mesh '+r'$\mathrm{Unet_{data}}$')
        ax2.set_aspect('equal')
        x1_label = ax2.get_xticklabels() 
        [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
        y1_label = ax2.get_yticklabels() 
        [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
        ax2.tick_params(axis='y',labelsize=10) 
        ax2.tick_params(axis='x',labelsize=10) 
        
        fig2.tight_layout(pad=1)
        fig2.savefig('result/mesh_irr_err.svg')
        # upx,upy,lowx,lowy,leftx,lefty,rightx,righty = plot_result.get_bound(self)
        # x, y = plot_result.plot_bound(self,upx,upy,lowx,lowy,leftx,lefty,rightx,righty)
        # myMesh = plot.plot_mesh()
        # F = plot_result.get_data(self,x,y)
        # u = plot_result.get_data(self,myMesh.x,myMesh.y)
        # F = torch.from_numpy(F).type(torch.FloatTensor).to(device)
        # net1 = UNetV2(in_channels=2, num_classes=2).to(device)
        # net2 = UNetV2(in_channels=2, num_classes=2).to(device)
        # net1.load_state_dict(torch.load("path/DeepLearning/best_mesh_phy.pth"))
        # net2.load_state_dict(torch.load("best_mesh.pth"))
        # with torch.no_grad():
        #     y1 = plot_result.get_result(self,net1(F),F)
        #     y2 = plot_result.get_result(self,net2(F),F)

        # fig1 = plt.figure()
        # ax1 = plt.subplot(2, 3, 1)
        # plt.imshow(u[0][0],cmap='coolwarm')
        # # plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal')
        # plt.axis('off')  #去掉坐标轴
        # ax1.set_title('label_x')
        # ax1.set_aspect('equal')
        # ax2 = plt.subplot(2, 3, 2)  
        # plt.imshow(y1[0][0].cpu().numpy(),cmap='coolwarm')
        # # plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal')
        # plt.axis('off')  #去掉坐标轴
        # ax2.set_title('phy-driven_pre_x')
        # ax2.set_aspect('equal')
        # ax3 = plt.subplot(2, 3, 3)  
        # plt.imshow(y2[0][0].cpu().numpy(),cmap='coolwarm')
        # # plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal')
        # plt.axis('off')  #去掉坐标轴
        # ax3.set_title('data-driven_pre_x')
        # ax3.set_aspect('equal')
        # plt.colorbar(aspect=50,ax=[ax1,ax2,ax3], pad=0.1,orientation='horizontal')
        # ax7 = plt.subplot(2, 3, 4)
        # plt.imshow(u[0][1],cmap='coolwarm')
        # # plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal')
        # plt.axis('off')  #去掉坐标轴
        # ax7.set_title('label_y')
        # ax7.set_aspect('equal')
        # ax8 = plt.subplot(2, 3, 5)  
        # plt.imshow(y1[0][1].cpu().numpy(),cmap='coolwarm')
        # # plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal')
        # plt.axis('off')  #去掉坐标轴
        # ax8.set_title('phy-driven_pre_y')
        # ax8.set_aspect('equal')
        # ax9 = plt.subplot(2, 3, 6)  
        # plt.imshow(y2[0][1].cpu().numpy(),cmap='coolwarm')
        # # plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal')
        # plt.axis('off')  #去掉坐标轴
        # ax9.set_title('data-driven_pre_y')
        # ax9.set_aspect('equal')
        # plt.colorbar(aspect=50,ax=[ax7,ax8,ax9], pad=0.1,orientation='horizontal')
        # fig1.savefig('result/mesh_reg.pdf')

        # fig2 = plt.figure()
        # ax4 = plt.subplot(2, 2, 1)  
        # plt.imshow(y1[0][0].cpu().numpy()-u[0][0],cmap='coolwarm')
        # # plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal')
        # plt.axis('off')  #去掉坐标轴
        # ax4.set_title('phy-driven_error_x')
        # ax4.set_aspect('equal')
        # ax5 = plt.subplot(2, 2, 2)  
        # plt.imshow(y2[0][0].cpu().numpy()-u[0][0],cmap='coolwarm')
        # # plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal')
        # plt.axis('off')  #去掉坐标轴
        # ax5.set_title('data-driven_error_x')
        # ax5.set_aspect('equal')
        # plt.colorbar(aspect=50,ax=[ax4,ax5], pad=0.1,orientation='horizontal')
        # ax10 = plt.subplot(2, 2, 3)  
        # plt.imshow(y1[0][1].cpu().numpy()-u[0][1],cmap='coolwarm')
        # # plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal')
        # plt.axis('off')  #去掉坐标轴
        # ax10.set_title('phy-driven_error_y')
        # ax10.set_aspect('equal')
        # ax11 = plt.subplot(2, 2, 4)  
        # plt.imshow(y2[0][1].cpu().numpy()-u[0][1],cmap='coolwarm')
        # # plt.colorbar(fraction=0.05, pad=0.05,orientation='horizontal')
        # plt.axis('off')  #去掉坐标轴
        # ax11.set_title('data-driven_error_y')
        # ax11.set_aspect('equal')
        # plt.colorbar(aspect=50,ax=[ax10,ax11], pad=0.1,orientation='horizontal')
        # fig2.savefig('result/mesh_reg_err.pdf')

        # fig3 = plt.figure()
        # ax2=plt.subplot(1,3,1)
        # plotMesh(ax2,u[0][0],u[0][1])
        # ax2.set_title('mesh_label')
        # ax2.set_aspect('equal')
        # ax2=plt.subplot(1,3,2)
        # plotMesh(ax2,y1[0][0].cpu(),y1[0][1].cpu())
        # ax2.set_title('mesh_phy-driven_pre')
        # ax2.set_aspect('equal')
        # ax2=plt.subplot(1,3,3)
        # plotMesh(ax2,y2[0][0].cpu(),y2[0][1].cpu())
        # ax2.set_title('mesh_data-driven_pre')
        # ax2.set_aspect('equal')
        # fig3.savefig('result/mesh_irr_err.pdf')


if __name__ == '__main__':
    plot = plot_result(256,64,10,50,100,0,20,level=4)

    # plot = plot_result(256,64,8,20,90,-10,10,level=4)
    # plot = plot_result(256,64,20,50,150,-10,50,level=4)

    # plot.plot_mesh()

    # upx,upy,lowx,lowy,leftx,lefty,rightx,righty = plot.get_bound()
    # plot.plot_bound(upx,upy,lowx,lowy,leftx,lefty,rightx,righty)

    # plot.plot_reg_result()

    # plot.plot_fd_result()

    # plot.plot_ml_result()

    # plot.plot_fdml_error()
    # plot.plot_singleml_error()

    # plot.plot_podbase(1)

    # plot.plot_podcal()

    plot.plot_multierror()

    # plot.plot_mlunet_error()

    # plot.plot_mlfno_error()

    # plot.plot_meshlearning()