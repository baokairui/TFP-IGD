import numpy as np
import scipy.io as sio
import bezier
import math
from pyMesh import hcubeMesh
from numpy.core.fromnumeric import size

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

# def data_generate(nx,ny,m,m1,m2,meshx,meshy,size,level):
#     n = 0
#     h = 0.01
#     left = np.linspace(10,16,3)
#     right = np.linspace(25,75,3)
#     middle = np.linspace(100,150,4)
#     y1 = np.linspace(0,30,8)
#     y2 = np.linspace(20,50,8)
#     for i in left:
#         for j in right:
#             for k in middle:
#                 for e in y1:
#                     for f in y2:
#                         size[n] = [i, j, k, e, f]
#                         print(n)
#                         upx,upy,lowx,lowy,leftx,lefty,rightx,righty = get_bound(e, f, nx, ny, k, i, j)
#                         myMesh=hcubeMesh(leftx,lefty,rightx,righty,lowx,lowy,upx,upy,h,True,True,tolMesh=1e-10,tolJoint=1)
#                         meshx[n] = myMesh.x
#                         meshy[n] = myMesh.y
#                         m1[n] = myMesh.dxdeta_ho ** 2 + myMesh.dydeta_ho ** 2
#                         m2[n] = myMesh.dxdxi_ho ** 2 + myMesh.dydxi_ho ** 2
#                         m[n] = (myMesh.dxdxi_ho ** 2 + myMesh.dydxi_ho ** 2) / (myMesh.dxdeta_ho ** 2 + myMesh.dydeta_ho ** 2)
#                         n = n + 1
#     sio.savemat('size',{'b':size})
#     sio.savemat('m'+str(level),{'b':m})
#     sio.savemat('meshx'+str(level),{'b':meshx})
#     sio.savemat('meshy'+str(level),{'b':meshy})
#     sio.savemat('m1'+str(level),{'b':m1})
#     sio.savemat('m2'+str(level),{'b':m2})

def data_generate(nx,ny,beta,alpha,gamma,J,meshx,meshy,level,para):
    h = 0.01
    # left = np.linspace(10,16,3)
    # right = np.linspace(25,75,3)
    # middle = np.linspace(100,150,4)
    # y1 = np.linspace(0,30,8)
    # y2 = np.linspace(20,50,8)
    for i,item in enumerate(para):
        # size[n] = [i, j, k, e, f]
        print(i)
        upx,upy,lowx,lowy,leftx,lefty,rightx,righty = get_bound(item[2], item[3], nx, ny, item[4], item[0], item[1])
        myMesh=hcubeMesh(leftx,lefty,rightx,righty,lowx,lowy,upx,upy,h,True,True,tolMesh=1e-10,tolJoint=1)
        meshx[i] = myMesh.x
        meshy[i] = myMesh.y
        alpha[i] = myMesh.dxdeta_ho ** 2 + myMesh.dydeta_ho ** 2
        gamma[i] = myMesh.dxdxi_ho ** 2 + myMesh.dydxi_ho ** 2
        beta[i] = myMesh.dxdxi_ho * myMesh.dxdeta_ho + myMesh.dydxi_ho * myMesh.dydeta_ho
        J[i] = myMesh.J_ho
    # sio.savemat('size',{'b':size})
    sio.savemat('alpha'+str(level),{'b':alpha})
    sio.savemat('meshx'+str(level),{'b':meshx})
    sio.savemat('meshy'+str(level),{'b':meshy})
    sio.savemat('gamma'+str(level),{'b':gamma})
    sio.savemat('beta'+str(level),{'b':beta})
    sio.savemat('J'+str(level),{'b':J})

def bound_generate(nx,ny,boundx,boundy,temp,para):
    for i,item in enumerate(para):
        # size[n] = [i, j, k, e, f]
        print(i)
        x=np.zeros([64,256])
        y=np.zeros([64,256])
        upx,upy,lowx,lowy,leftx,lefty,rightx,righty = get_bound(item[2], item[3], nx, ny, item[4], item[0], item[1])           
        x[:,0]=leftx; y[:,0]=lefty
        x[:,-1]=rightx; y[:,-1]=righty
        x[0,:]=lowx; y[0,:]=lowy
        x[-1,:]=upx; y[-1,:]=upy
        boundx[i] = x; boundy[i] = y  
    sio.savemat('boundx'+str(temp),{'b':boundx})
    sio.savemat('boundy'+str(temp),{'b':boundy})


if __name__ == '__main__':
    level = 4
    max_nx = 256; max_ny = 64
    para = sio.loadmat("./dataset/para-2500.mat")['b']
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

