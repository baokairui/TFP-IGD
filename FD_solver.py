from math import gamma
from google.protobuf.symbol_database import Default
import numpy as np
import scipy
import scipy.io as sio
from scipy.sparse.linalg import spsolve
from ML_mesh_parameter import get_bound
from ML_regular_temperature import get_A,get_b
from pyMesh import hcubeMesh


class solver(object):
    def __init__(self,nx,ny,left_bound,right_bound,x_bound,y1,y2,h=0.01):
        self.nx = nx
        self.ny = ny
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.x_bound = x_bound
        self.y1 = y1
        self.y2 = y2
        self.h = h
    def mesh(self):
        upx,upy,lowx,lowy,leftx,lefty,rightx,righty = get_bound(self.y1, self.y2, self.nx, self.ny, self.x_bound, self.left_bound, self.right_bound)
        myMesh=hcubeMesh(leftx,lefty,rightx,righty,lowx,lowy,upx,upy,self.h,True,True,tolMesh=1e-10,tolJoint=1)
        gamma = (myMesh.dydxi_ho ** 2 + myMesh.dxdxi_ho ** 2) 
        alpha = (myMesh.dydeta_ho ** 2 + myMesh.dxdeta_ho ** 2)
        return myMesh,alpha,gamma
    def get_result(self):
        myMesh,m1,m2 = solver.mesh(self)
        A = get_A(self.nx,self.ny,m1,m2)
        b = get_b(self.nx,self.ny)
        t = spsolve(A, b)
        tem = t.reshape(self.ny,self.nx)
        return tem,myMesh

if __name__ == '__main__':
    fd = solver(32,8,10,50,150,50,0)
    result = fd.get_result()[0]



