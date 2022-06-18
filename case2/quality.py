import numpy as np
import scipy.io as sio
import math
from pyMesh import hcubeMesh,plotMesh,visualize2D
from ML_mesh_parameter import get_bezier
import matplotlib.pyplot as plt
import torch
from model.Unet.unet import UNetV2
from matplotlib import rcParams
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = {
    "font.family":'serif',
    "font.size": 20,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)
plt.rc('font', family='Times New Roman')

# idx = 365

def mesh(coordinate):
    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8 = coordinate
    upx, upy = get_bezier(6,64,[x1,x2,x3,0,-x3,-x2,-x1],[y1,y2,y3,y3,y3,y2,y1])
    leftx, lefty = get_bezier(3,64,[x5,x5,x4,x1],[0,y5,y4,y1])
    rightx, righty = -leftx, lefty
    lowx, lowy = get_bezier(8,64,[x5,x6,x7,x8,0,-x8,-x7,-x6,-x5],[0,y6,y7,y8,y8,y8,y7,y6,0])
    myMesh = hcubeMesh(leftx,lefty,rightx,righty,lowx,lowy,upx,upy,0.01,True,True,saveDir='result/mesh.pdf',tolMesh=1e-10,tolJoint=1)
    return upx,upy,lowx,lowy,leftx,lefty,rightx,righty,myMesh

def plot_bound(upx,upy,lowx,lowy,leftx,lefty,rightx,righty):
        x = np.zeros([64,64])
        y = np.zeros([64,64])
        plt.figure(figsize=(2.2,1))
        x[:,0] = leftx; y[:,0] = lefty
        x[:,-1] = rightx; y[:,-1] = righty
        x[0,:] = lowx; y[0,:] = lowy
        x[-1,:] = upx; y[-1,:] = upy
        return x,y

# upx,upy,lowx,lowy,leftx,lefty,rightx,righty,myMesh = mesh([-99.75 ,  82.85 , -44.95 , 103.45 , -31.11 ,  67.09 , -70.29 ,
#         73.83 , -64.175,  36.65 , -48.725,  18.555, -25.045, -42.105,
#        -12.565,  18.615])
upx,upy,lowx,lowy,leftx,lefty,rightx,righty,myMesh = mesh([-100,90,-45,105,-30,60,-75,75,-65,40,-45,15,-25,-40,-15,20])
       
boundx, boundy = plot_bound(upx,upy,lowx,lowy,leftx,lefty,rightx,righty)


# labelx = sio.loadmat('./dataset/2000/meshx/meshx1')['b'].reshape(2000,64,64)
# labely = sio.loadmat('./dataset/2000/meshy/meshy1')['b'].reshape(2000,64,64)
# boundx = sio.loadmat('./dataset/2000/bound/boundx1')['b'].reshape(2000,64,64)
# boundy = sio.loadmat('./dataset/2000/bound/boundy1')['b'].reshape(2000,64,64)

def cal_angle(point_a, point_b, point_c):
    a_x, b_x, c_x = point_a[0], point_b[0], point_c[0]  # 点a、b、c的x坐标
    a_y, b_y, c_y = point_a[1], point_b[1], point_c[1]  # 点a、b、c的y坐标

    if len(point_a) == len(point_b) == len(point_c) == 3:
        # print("坐标点为3维坐标形式")
        a_z, b_z, c_z = point_a[2], point_b[2], point_c[2]  # 点a、b、c的z坐标
    else:
        a_z, b_z, c_z = 0,0,0  # 坐标点为2维坐标形式，z 坐标默认值设为0
        # print("坐标点为2维坐标形式，z 坐标默认值设为0")

    # 向量 m=(x1,y1,z1), n=(x2,y2,z2)
    x1,y1,z1 = (a_x-b_x),(a_y-b_y),(a_z-b_z)
    x2,y2,z2 = (c_x-b_x),(c_y-b_y),(c_z-b_z)

    # 两个向量的夹角，即角点b的夹角余弦值
    cos_b = (x1*x2 + y1*y2 + z1*z2) / (math.sqrt(x1**2 + y1**2 + z1**2) *(math.sqrt(x2**2 + y2**2 + z2**2))) # 角点b的夹角余弦值
    B = math.degrees(math.acos(cos_b)) # 角点b的夹角值
    return B

def get_data(data1,data2):
    data = np.zeros((1,2,64,64))
    data[0][0] = data1
    data[0][1] = data2
    return data

def get_result(output, data):
        result = data.clone()
        result[...,1:-1,1:-1] = output[...,1:-1,1:-1]
        return result

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

bx1 = myMesh.x[:-1,:-1].reshape(-1)
by1 = myMesh.y[:-1,:-1].reshape(-1)
ax1 = myMesh.x[:-1,1:].reshape(-1)
ay1 = myMesh.y[:-1,1:].reshape(-1)
cx1 = myMesh.x[1:,:-1].reshape(-1)
cy1 = myMesh.y[1:,:-1].reshape(-1)

bx2 = myMesh.x[:-1,1:].reshape(-1)
by2 = myMesh.y[:-1,1:].reshape(-1)
ax2 = myMesh.x[:-1,:-1].reshape(-1)
ay2 = myMesh.y[:-1,:-1].reshape(-1)
cx2 = myMesh.x[1:,1:].reshape(-1)
cy2 = myMesh.y[1:,1:].reshape(-1)

bx3 = myMesh.x[1:,:-1].reshape(-1)
by3 = myMesh.y[1:,:-1].reshape(-1)
ax3 = myMesh.x[:-1,:-1].reshape(-1)
ay3 = myMesh.y[:-1,:-1].reshape(-1)
cx3 = myMesh.x[1:,1:].reshape(-1)
cy3 = myMesh.y[1:,1:].reshape(-1)

bx4 = myMesh.x[1:,1:].reshape(-1)
by4 = myMesh.y[1:,1:].reshape(-1)
ax4 = myMesh.x[:-1,1:].reshape(-1)
ay4 = myMesh.y[:-1,1:].reshape(-1)
cx4 = myMesh.x[1:,:-1].reshape(-1)
cy4 = myMesh.y[1:,:-1].reshape(-1)

angle1 = np.array([])
angle2 = np.array([])
angle3 = np.array([])
angle4 = np.array([])
for i in range (bx1.shape[0]):
    print(i)
    angle1 = np.append(angle1,cal_angle([ax1[i],ay1[i]],[bx1[i],by1[i]],[cx1[i],cy1[i]]))
    angle2 = np.append(angle2,cal_angle([ax2[i],ay2[i]],[bx2[i],by2[i]],[cx2[i],cy2[i]]))
    angle3 = np.append(angle3,cal_angle([ax3[i],ay3[i]],[bx3[i],by3[i]],[cx3[i],cy3[i]]))
    angle4 = np.append(angle4,cal_angle([ax4[i],ay4[i]],[bx4[i],by4[i]],[cx4[i],cy4[i]]))

skewness = np.array([])
thetamax = np.array([])
for i in range (angle1.shape[0]):
    thetam = max(angle1[i],angle2[i],angle3[i],angle4[i])
    thetamin = min(angle1[i],angle2[i],angle3[i],angle4[i])
    sk1 = (thetam - 90) / (180 - 90)
    sk2 = (90 - thetamin) / 90
    skewness = np.append(skewness,max(sk1,sk2))
    thetamax = np.append(thetamax,thetam)

F = get_data(boundx,boundy)
u = get_data(myMesh.x,myMesh.y)
F = torch.from_numpy(F).type(torch.FloatTensor).to(device)
net1 = UNetV2(in_channels=2, num_classes=2).to(device)
net2 = UNetV2(in_channels=2, num_classes=2).to(device)
net1.load_state_dict(torch.load("path/Mesh/best_mesh_phyaccnorm2000.pth"))
net2.load_state_dict(torch.load("path/Mesh/best_mesh_data.pth"))
with torch.no_grad():
    y1 = get_result_norm(net1,F)
    y2 = get_result(net2(F),F)

prex = y1[0][0].cpu().numpy()
prey = y1[0][1].cpu().numpy()

bx1 = prex[:-1,:-1].reshape(-1)
by1 = prey[:-1,:-1].reshape(-1)
ax1 = prex[:-1,1:].reshape(-1)
ay1 = prey[:-1,1:].reshape(-1)
cx1 = prex[1:,:-1].reshape(-1)
cy1 = prey[1:,:-1].reshape(-1)

bx2 = prex[:-1,1:].reshape(-1)
by2 = prey[:-1,1:].reshape(-1)
ax2 = prex[:-1,:-1].reshape(-1)
ay2 = prey[:-1,:-1].reshape(-1)
cx2 = prex[1:,1:].reshape(-1)
cy2 = prey[1:,1:].reshape(-1)

bx3 = prex[1:,:-1].reshape(-1)
by3 = prey[1:,:-1].reshape(-1)
ax3 = prex[:-1,:-1].reshape(-1)
ay3 = prey[:-1,:-1].reshape(-1)
cx3 = prex[1:,1:].reshape(-1)
cy3 = prey[1:,1:].reshape(-1)

bx4 = prex[1:,1:].reshape(-1)
by4 = prey[1:,1:].reshape(-1)
ax4 = prex[:-1,1:].reshape(-1)
ay4 = prey[:-1,1:].reshape(-1)
cx4 = prex[1:,:-1].reshape(-1)
cy4 = prey[1:,:-1].reshape(-1)

anglep1 = np.array([])
anglep2 = np.array([])
anglep3 = np.array([])
anglep4 = np.array([])
for i in range (bx1.shape[0]):
    print(i)
    anglep1 = np.append(anglep1,cal_angle([ax1[i],ay1[i]],[bx1[i],by1[i]],[cx1[i],cy1[i]]))
    anglep2 = np.append(anglep2,cal_angle([ax2[i],ay2[i]],[bx2[i],by2[i]],[cx2[i],cy2[i]]))
    anglep3 = np.append(anglep3,cal_angle([ax3[i],ay3[i]],[bx3[i],by3[i]],[cx3[i],cy3[i]]))
    anglep4 = np.append(anglep4,cal_angle([ax4[i],ay4[i]],[bx4[i],by4[i]],[cx4[i],cy4[i]]))

skewnessp = np.array([])
thetamaxp = np.array([])
for i in range (anglep1.shape[0]):
    thetam = max(anglep1[i],anglep2[i],anglep3[i],anglep4[i])
    thetamin = min(anglep1[i],anglep2[i],anglep3[i],anglep4[i])
    sk1 = (thetam - 90) / (180 - 90)
    sk2 = (90 - thetamin) / 90
    skewnessp = np.append(skewnessp,max(sk1,sk2))
    thetamaxp = np.append(thetamaxp,thetam)

plt.figure(figsize=(12,6))
ax=plt.subplot(1,3,1)
visualize2D(ax,myMesh.x[1:,1:],myMesh.y[1:,1:],thetamax,'horizontal',[90,135],fraction=0.06, pad=0.05)
plt.tick_params(labelsize=12, rotation=0)
ax.set_title('Reference ' + r'$\theta_{max}$',fontdict={'family' : 'Times New Roman','size': 15})
ax.set_aspect('equal')
ax=plt.subplot(1,3,2)
visualize2D(ax,myMesh.x[1:,1:],myMesh.y[1:,1:],thetamaxp,'horizontal',[90,135],fraction=0.06, pad=0.05)
plt.tick_params(labelsize=12, rotation=0)
ax.set_title(r'$Unet_{phy}$' + r' $\theta_{max}$',fontdict={'family' : 'Times New Roman','size': 15})
ax.set_aspect('equal')
ax=plt.subplot(1,3,3)
visualize2D(ax,myMesh.x[1:,1:],myMesh.y[1:,1:],thetamax-thetamaxp,'horizontal',[-0.5,0.5],fraction=0.06, pad=0.05)
plt.tick_params(labelsize=12, rotation=0)
ax.set_title('error',fontdict={'family' : 'Times New Roman','size': 15})
ax.set_aspect('equal')
plt.savefig('./result/qualitytheta.pdf',bbox_inches='tight')

plt.figure(figsize=(12,6))
ax=plt.subplot(1,3,1)
visualize2D(ax,myMesh.x[1:,1:],myMesh.y[1:,1:],skewness,'horizontal',[0,0.5],fraction=0.06, pad=0.05)
plt.tick_params(labelsize=12, rotation=0)
ax.set_title('Reference Skew',fontdict={'family' : 'Times New Roman','size': 15})
ax.set_aspect('equal')
ax=plt.subplot(1,3,2)
visualize2D(ax,myMesh.x[1:,1:],myMesh.y[1:,1:],skewnessp,'horizontal',[0,0.5],fraction=0.06, pad=0.05)
plt.tick_params(labelsize=12, rotation=0)
ax.set_title(r'$\mathrm{Unet_{phy}}$' + ' Skew',fontdict={'family' : 'Times New Roman','size': 15})
ax.set_aspect('equal')
ax=plt.subplot(1,3,3)
visualize2D(ax,myMesh.x[1:,1:],myMesh.y[1:,1:],skewness-skewnessp,'horizontal',[-0.005,0.005],fraction=0.06, pad=0.05)
plt.tick_params(labelsize=12, rotation=0)
ax.set_title('error',fontdict={'family' : 'Times New Roman','size': 15})
ax.set_aspect('equal')
plt.savefig('./result/qualityskewness.pdf',bbox_inches='tight')



print(np.abs(np.mean(skewness-skewnessp)))
print(np.abs(np.mean(thetamax-thetamaxp)))