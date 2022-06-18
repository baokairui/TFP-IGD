import numpy as np
import scipy.io as sio
import bezier
import math
from pyMesh import hcubeMesh


def get_bezier(order,number,x,y):
    nodes = np.asfortranarray([x,y])
    curve = bezier.Curve(nodes, degree=order)
    s_vals = np.linspace(0.0, 1.0, number)
    data=curve.evaluate_multi(s_vals)
    return data[0],data[1]

def get_bound(nx, ny, coordinate):
    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8 = coordinate
    upx, upy = get_bezier(6,nx,[x1,x2,x3,0,-x3,-x2,-x1],[y1,y2,y3,y3,y3,y2,y1])
    leftx, lefty = get_bezier(3,ny,[x5,x5,x4,x1],[0,y5,y4,y1])
    rightx, righty = -leftx, lefty
    lowx, lowy = get_bezier(8,nx,[x5,x6,x7,x8,0,-x8,-x7,-x6,-x5],[0,y6,y7,y8,y8,y8,y7,y6,0])
    return upx,upy,lowx,lowy,leftx,lefty,rightx,righty

def data_generate(nx,ny,beta,alpha,gamma,J,meshx,meshy,level,para):
    h = 0.01
    for i,item in enumerate(para):
        print(i)
        upx,upy,lowx,lowy,leftx,lefty,rightx,righty = get_bound(nx,ny,item)
        myMesh=hcubeMesh(leftx,lefty,rightx,righty,lowx,lowy,upx,upy,h,True,True,tolMesh=1e-10,tolJoint=1)
        meshx[i] = myMesh.x
        meshy[i] = myMesh.y
        alpha[i] = myMesh.dxdeta_ho ** 2 + myMesh.dydeta_ho ** 2
        gamma[i] = myMesh.dxdxi_ho ** 2 + myMesh.dydxi_ho ** 2
        beta[i] = myMesh.dxdxi_ho * myMesh.dxdeta_ho + myMesh.dydxi_ho * myMesh.dydeta_ho
        J[i] = myMesh.J_ho
    sio.savemat('alpha'+str(level),{'b':alpha})
    sio.savemat('meshx'+str(level),{'b':meshx})
    sio.savemat('meshy'+str(level),{'b':meshy})
    sio.savemat('gamma'+str(level),{'b':gamma})
    sio.savemat('beta'+str(level),{'b':beta})
    sio.savemat('J'+str(level),{'b':J})

def bound_generate(nx,ny,boundx,boundy,temp,para):
    for i,item in enumerate(para):
        print(i)
        x=np.zeros([64,64])
        y=np.zeros([64,64])
        upx,upy,lowx,lowy,leftx,lefty,rightx,righty = get_bound(nx,ny,item)        
        x[:,0]=leftx; y[:,0]=lefty
        x[:,-1]=rightx; y[:,-1]=righty
        x[0,:]=lowx; y[0,:]=lowy
        x[-1,:]=upx; y[-1,:]=upy
        boundx[i] = x; boundy[i] = y  
    sio.savemat('boundx'+str(temp),{'b':boundx})
    sio.savemat('boundy'+str(temp),{'b':boundy})


if __name__ == '__main__':
    level = 4
    max_nx = 64; max_ny = 64
    para = sio.loadmat("./dataset/para-1500.mat")['b']
    lenth = len(para)
    for i in range(level):
        nx = math.ceil(max_nx / 2 ** i)
        ny = math.ceil(max_ny / 2 ** i)
        # size = np.empty((lenth,5))
        beta = np.empty((lenth,ny,nx))
        alpha = np.empty((lenth,ny,nx))
        gamma = np.empty((lenth,ny,nx))
        J = np.empty((lenth,ny,nx))
        meshx = np.empty((lenth,ny,nx))
        meshy = np.empty((lenth,ny,nx))
        data_generate(nx,ny,beta,alpha,gamma,J,meshx,meshy,i+1,para)
    boundx = np.empty((lenth,max_ny,max_nx))
    boundy = np.empty((lenth,max_ny,max_nx))
    bound_generate(max_nx,max_ny,boundx,boundy,1,para)

