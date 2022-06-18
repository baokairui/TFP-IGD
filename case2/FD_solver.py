from math import gamma
from google.protobuf.symbol_database import Default
import numpy as np
import scipy
import scipy.io as sio
from scipy.sparse.linalg import spsolve
from ML_mesh_parameter import get_bound
from ML_regular_temperature_neuman import get_A,get_b
from pyMesh import hcubeMesh


class solver(object):
    def __init__(self,nx,ny,coordinate,h=0.01):
        self.nx = nx
        self.ny = ny
        self.coordinate = coordinate
        self.h = h
    def mesh(self):
        upx,upy,lowx,lowy,leftx,lefty,rightx,righty = get_bound(self.nx, self.ny, self.coordinate)
        myMesh=hcubeMesh(leftx,lefty,rightx,righty,lowx,lowy,upx,upy,self.h,True,True,tolMesh=1e-10,tolJoint=1)
        gamma = (myMesh.dydxi_ho ** 2 + myMesh.dxdxi_ho ** 2) 
        alpha = (myMesh.dydeta_ho ** 2 + myMesh.dxdeta_ho ** 2)
        beta = myMesh.dxdxi_ho * myMesh.dxdeta_ho + myMesh.dydxi_ho * myMesh.dydeta_ho
        return myMesh,alpha,gamma,beta
    def get_result(self):
        myMesh,alpha,gamma,beta = solver.mesh(self)
        n = 2 * self.h * myMesh.J_ho / np.sqrt(gamma)
        A = get_A(self.nx,self.ny,alpha,beta,gamma,beta/gamma)
        b = get_b(self.nx,self.ny,1,1,n)
        t = spsolve(A, b)
        tem = t.reshape(self.ny,self.nx)
        return tem,myMesh

if __name__ == '__main__':
    fd = solver(64,64,[-93.3925 ,  83.6225 , -48.1875 , 107.0675 , -37.1575 ,  67.8275 ,
       -72.4975 ,  74.8075 , -61.65375,  30.4425 , -49.41375,  18.27125,
       -28.24375, -43.83125, -11.02875,  15.42625])
    result = fd.get_result()[0]
    print(np.max(result))



