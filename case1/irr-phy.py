import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import bezier
from torch.utils.data import DataLoader,Dataset
from pyMesh import hcubeMesh, visualize2D,setAxisLabel,to4DTensor
from sklearn.metrics import mean_squared_error as calMAE
from scipy.interpolate import Rbf
from torch.optim import lr_scheduler
from model.Unet.unetphy import UNetV2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_label = sio.loadmat("./dataset/T/T1.mat")['b'][0].reshape(64,256)

def get_bezier(y1,y2,nx,x_bound,left_bound,right_bound):
    # middle = (left_bound + right_bound) / 2
    nodes = np.asfortranarray([[-x_bound,-50,100,x_bound],[left_bound,y1,y2,right_bound]])
    curve = bezier.Curve(nodes, degree=3)
    s_vals = np.linspace(0.0, 1.0, nx)
    data=curve.evaluate_multi(s_vals)
    return data[0],data[1]

def get_bound(x1, x2, nx, ny, x_bound, left_bound, right_bound):
    upx,upy = get_bezier(x1,x2,nx,x_bound,left_bound,right_bound)
    lowy = -upy
    lowx = upx
    lefty = np.linspace(-left_bound,left_bound,ny)
    righty = np.linspace(-right_bound,right_bound,ny)
    leftx = np.ones_like(lefty) * -x_bound
    rightx = np.ones_like(righty)* x_bound
    return upx,upy,lowx,lowy,leftx,lefty,rightx,righty

class VaryGeoDataset(Dataset):
	"""docstring for hcubeMeshDataset"""
	def __init__(self,MeshList):
		self.MeshList=MeshList
	def __len__(self):
		return len(self.MeshList)
	def __getitem__(self,idx):
		mesh=self.MeshList[idx]
		x=mesh.x
		y=mesh.y
		xi=mesh.xi
		eta=mesh.eta
		J=mesh.J_ho
		Jinv=mesh.Jinv_ho
		dxdxi=mesh.dxdxi_ho
		dydxi=mesh.dydxi_ho
		dxdeta=mesh.dxdeta_ho
		dydeta=mesh.dydeta_ho
		cord=np.zeros([2,x.shape[0],x.shape[1]])
		cord[0,:,:]=x; cord[1,:,:]=y
		InvariantInput=np.zeros([2,J.shape[0],J.shape[1]])
		InvariantInput[0,:,:]=J
		InvariantInput[1,:,:]=Jinv
		return [InvariantInput,cord,xi,eta,J,
		        Jinv,dxdxi,dydxi,
		        dxdeta,dydeta]

upx,upy,lowx,lowy,leftx,lefty,rightx,righty = get_bound(0, 50, 256, 64, 150, 10, 50)
h=0.01
ny=len(leftx);nx=len(lowx)
myMesh=hcubeMesh(leftx,lefty,rightx,righty,
	             lowx,lowy,upx,upy,h,True,True,
	             tolMesh=1e-10,tolJoint=1)

batchSize=1
NvarInput=2
NvarOutput=1
nEpochs=3000
Ns=1
nu=0.03
net = UNetV2(nx, ny, in_channels=2, num_classes=1).to(device)
# model=USCNNSep(h,nx,ny,NvarInput,NvarOutput).to('cuda')
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=50, verbose=False)
# optimizer = optim.Adam(model.parameters(),lr=lr)
padSingleSide=1
udfpad=nn.ConstantPad2d([padSingleSide,padSingleSide,padSingleSide,padSingleSide],0)
MeshList=[]
MeshList.append(myMesh)
train_set=VaryGeoDataset(MeshList)
training_data_loader=DataLoader(dataset=train_set,
	                            batch_size=batchSize)
OFV_sb=data_label

def dfdx(f,dydeta,dydxi,Jinv):
	dfdxi_internal=(-f[:,:,:,4:]+8*f[:,:,:,3:-1]-8*f[:,:,:,1:-3]+f[:,:,:,0:-4])/12/h 	
	dfdxi_left=(-11*f[:,:,:,0:-3]+18*f[:,:,:,1:-2]-9*f[:,:,:,2:-1]+2*f[:,:,:,3:])/6/h
	dfdxi_right=(11*f[:,:,:,3:]-18*f[:,:,:,2:-1]+9*f[:,:,:,1:-2]-2*f[:,:,:,0:-3])/6/h
	dfdxi=torch.cat((dfdxi_left[:,:,:,0:2],dfdxi_internal,dfdxi_right[:,:,:,-2:]),3)
	dfdeta_internal=(-f[:,:,4:,:]+8*f[:,:,3:-1,:]-8*f[:,:,1:-3,:]+f[:,:,0:-4,:])/12/h	
	dfdeta_low=(-11*f[:,:,0:-3,:]+18*f[:,:,1:-2,:]-9*f[:,:,2:-1,:]+2*f[:,:,3:,:])/6/h
	dfdeta_up=(11*f[:,:,3:,:]-18*f[:,:,2:-1,:]+9*f[:,:,1:-2,:]-2*f[:,:,0:-3,:])/6/h
	dfdeta=torch.cat((dfdeta_low[:,:,0:2,:],dfdeta_internal,dfdeta_up[:,:,-2:,:]),2)
	dfdx=Jinv*(dfdxi*dydeta-dfdeta*dydxi)
	return dfdx
	
def dfdy(f,dxdxi,dxdeta,Jinv):
	dfdxi_internal=(-f[:,:,:,4:]+8*f[:,:,:,3:-1]-8*f[:,:,:,1:-3]+f[:,:,:,0:-4])/12/h	
	dfdxi_left=(-11*f[:,:,:,0:-3]+18*f[:,:,:,1:-2]-9*f[:,:,:,2:-1]+2*f[:,:,:,3:])/6/h
	dfdxi_right=(11*f[:,:,:,3:]-18*f[:,:,:,2:-1]+9*f[:,:,:,1:-2]-2*f[:,:,:,0:-3])/6/h
	dfdxi=torch.cat((dfdxi_left[:,:,:,0:2],dfdxi_internal,dfdxi_right[:,:,:,-2:]),3)
	
	dfdeta_internal=(-f[:,:,4:,:]+8*f[:,:,3:-1,:]-8*f[:,:,1:-3,:]+f[:,:,0:-4,:])/12/h	
	dfdeta_low=(-11*f[:,:,0:-3,:]+18*f[:,:,1:-2,:]-9*f[:,:,2:-1,:]+2*f[:,:,3:,:])/6/h
	dfdeta_up=(11*f[:,:,3:,:]-18*f[:,:,2:-1,:]+9*f[:,:,1:-2,:]-2*f[:,:,0:-3,:])/6/h
	dfdeta=torch.cat((dfdeta_low[:,:,0:2,:],dfdeta_internal,dfdeta_up[:,:,-2:,:]),2)
	dfdy=Jinv*(dfdeta*dxdxi-dfdxi*dxdeta)
	return dfdy
# def dfdx(f,dydeta,dydxi,Jinv):
# 	dfdxi_internal=(-f[:,:,:,4:]+8*f[:,:,:,3:-1]-8*f[:,:,:,1:-3]+f[:,:,:,0:-4])/12/h 	
# 	dfdxi_left=(f[:,:,:,1:]-f[:,:,:,0:-1])/h
# 	dfdxi_right=(f[:,:,:,:-1]-f[:,:,:,1:])/h
# 	dfdxi=torch.cat((dfdxi_left[:,:,:,0:2],dfdxi_internal,dfdxi_right[:,:,:,-2:]),3)
# 	dfdeta_internal=(-f[:,:,4:,:]+8*f[:,:,3:-1,:]-8*f[:,:,1:-3,:]+f[:,:,0:-4,:])/12/h	
# 	dfdeta_low=(f[:,:,1:,:]-f[:,:,0:-1,:])/h
# 	dfdeta_up=(f[:,:,:-1,:]-f[:,:,:,1:])/h
# 	dfdeta=torch.cat((dfdeta_low[:,:,0:2,:],dfdeta_internal,dfdeta_up[:,:,-2:,:]),2)
# 	dfdx=Jinv*(dfdxi*dydeta-dfdeta*dydxi)
# 	return dfdx
def train(epoch):
	# startTime=time.time()
	xRes=0
	yRes=0
	mRes=0
	eU=0
	eV=0
	eP=0
	for iteration, batch in enumerate(training_data_loader):
		[JJInv,coord,xi,eta,J,Jinv,dxdxi,dydxi,dxdeta,dydeta]=to4DTensor(batch)
		optimizer.zero_grad()
		output=net(coord)
		output_pad=udfpad(output)
		outputV=output_pad[:,0,:,:].reshape(output_pad.shape[0],1,
			                                output_pad.shape[2],
			                                output_pad.shape[3])
		for j in range(batchSize):
			outputV[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=22
			outputV[j,0,:padSingleSide,padSingleSide:-padSingleSide]=60				   		
			outputV[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=60					    			
			outputV[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=60
			outputV[j,0,0,0]=0.5*(outputV[j,0,0,1]+outputV[j,0,1,0])
			outputV[j,0,0,-1]=0.5*(outputV[j,0,0,-2]+outputV[j,0,1,-1]) 
			# outputV[j,0,0,0]=22
			# outputV[j,0,-1,0]=22					    
			# outputV[j,0,0,-1]=0.5*(outputV[j,0,0,-2]+outputV[j,0,-1,-2])
			# outputV[j,0,-1,-1]=0.5*(outputV[j,0,-1,-2]+outputV[j,0,-2,-1])
		dvdx=dfdx(outputV,dydeta,dydxi,Jinv)
		d2vdx2=dfdx(dvdx,dydeta,dydxi,Jinv)
		dvdy=dfdy(outputV,dxdxi,dxdeta,Jinv)
		d2vdy2=dfdy(dvdy,dxdxi,dxdeta,Jinv)
		continuity=(d2vdy2+d2vdx2);
		loss=criterion(continuity,continuity*0)
		loss.backward()
		optimizer.step()
		loss_mass=criterion(continuity, continuity*0)
		mRes+=loss_mass.item()
		CNNVNumpy=outputV[0,0,:,:].cpu().detach().numpy()
		eV=eV+np.sqrt(calMAE(OFV_sb,CNNVNumpy)/calMAE(OFV_sb,OFV_sb*0))
	print('Epoch is ',epoch)
	print("mRes Loss is", (mRes/len(training_data_loader)))
	print("eV Loss is", (eV/len(training_data_loader)))
	if epoch%5000==0 or epoch%nEpochs==0 or np.sqrt(calMAE(OFV_sb,CNNVNumpy)/calMAE(OFV_sb,OFV_sb*0))<0.01:
		torch.save(net, str(epoch)+'.pth')
		fig1=plt.figure()
		ax=plt.subplot(1,3,1)
		visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           outputV[0,0,1:-1,1:-1].cpu().detach().numpy(),'horizontal',[22,60])
		# setAxisLabel(ax,'p')
		ax.set_title('CNN '+r'$T$')
		ax.set_aspect('equal')
		ax=plt.subplot(1,3,2)
		visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           OFV_sb[1:-1,1:-1],'horizontal',[22,60])
		# setAxisLabel(ax,'p')
		ax.set_aspect('equal')
		ax.set_title('FV '+r'$T$')
		ax=plt.subplot(1,3,3)
		visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           outputV[0,0,1:-1,1:-1].cpu().detach().numpy()-OFV_sb[1:-1,1:-1],'horizontal',[-1,1])
		# setAxisLabel(ax,'p')
		ax.set_title('error phy')
		ax.set_aspect('equal')
		fig1.tight_layout(pad=1)
		fig1.savefig('result/irr-phyT.pdf',bbox_inches='tight')
		plt.close(fig1)		
	return (mRes/len(training_data_loader)),(eV/len(training_data_loader))
# MRes=[]
# EV=[]
# TotalstartTime=time.time()
for epoch in range(1,nEpochs+1):
	mres,ev=train(epoch)
	# MRes.append(mres)
	# EV.append(ev)
	scheduler.step(mres)
	if ev<0.01:
		break
# TimeSpent=time.time()-TotalstartTime
# plt.figure()
# plt.plot(MRes,'-*',label='Equation Residual')
# plt.xlabel('Epoch')
# plt.ylabel('Residual')
# plt.legend()
# plt.yscale('log')
# plt.savefig('convergence.pdf',bbox_inches='tight')
# # tikzplotlib.save('convergence.tikz')
# plt.figure()
# plt.plot(EV,'-x',label=r'$e_v$')
# plt.xlabel('Epoch')
# plt.ylabel('Error')
# plt.legend()
# plt.yscale('log')
# plt.savefig('error.pdf',bbox_inches='tight')
# # tikzplotlib.save('error.tikz')
# EV=np.asarray(EV)
# MRes=np.asarray(MRes)
# np.savetxt('EV.txt',EV)
# np.savetxt('MRes.txt',MRes)
# np.savetxt('TimeSpent.txt',np.zeros([2,2])+TimeSpent)