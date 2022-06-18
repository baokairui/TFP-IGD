from math import gamma
from unicodedata import name
import numpy as np
import scipy
import scipy.io as sio
from scipy.sparse.linalg import spsolve


def get_A(nx,ny,a,b,r):
    a = a[1:-1,1:-1]
    b = b[1:-1,1:-1]
    r = r[1:-1,1:-1]
    a = np.pad(a,((1,1),(1,1)),'constant',constant_values=1).reshape(-1,)
    b = np.pad(b,((1,1),(1,1)),'constant',constant_values=1).reshape(-1,)
    r = np.pad(r,((1,1),(1,1)),'constant',constant_values=1).reshape(-1,)

    d0 = - 2 * (a + r)

    d11 = np.zeros(nx)
    d12 = np.ones(nx)
    d12[0] = 0
    d12[-1] = 0
    d12 = np.tile(d12, ny-2)
    d13 = np.zeros(nx-1)
    d1 = np.concatenate((d11,d12,d13),axis=0)
    d1 = d1 * a[:-1]
    d1_ = np.concatenate((d13,d12,d11),axis=0)
    d1_ = d1_ * a[1:]

    d21 = np.zeros(nx)
    d22 = np.ones(nx) * 0.5
    d22[0] = 0
    d22[-1] = 0
    d22 = np.tile(d22, ny-2)
    d23 = np.zeros(1)
    d2 = np.concatenate((d21,d22,d23),axis=0)
    d2 = d2 * b[:-nx+1]
    d2_ = np.concatenate((d23,d22,d21),axis=0)
    d2_ = d2_ * b[nx-1:]

    d31 = np.zeros(nx)
    d32 = np.ones(nx)
    d32[0] = 0
    d32[-1] = 0
    d32 = np.tile(d32, ny-2)
    d3 = np.concatenate((d31,d32),axis=0)
    d3 = d3 * r[:-nx]
    d3_ = np.concatenate((d32,d31),axis=0)
    d3_ = d3_ * r[nx:]


    d41 = np.zeros(nx)
    d42 = - np.ones(nx) * 0.5
    d42[0] = 0
    d42[-1] = 0
    d42 = np.tile(d42, ny-3)
    d43 = - np.ones(nx-1) * 0.5
    d43[0] = 0
    d4 = np.concatenate((d41,d42,d43),axis=0)
    d4 = d4 * b[:-nx-1]
    d43_ = - np.ones(nx-1) * 0.5
    d43_[-1] = 0
    d4_ = np.concatenate((d43_,d42,d41),axis=0)
    d4_ = d4_ * b[nx+1:]

    A = scipy.sparse.diags([d0, d1, d1_, d2, d2_, d3, d3_, d4, d4_], [0, 1, -1, nx-1, -nx+1, nx, - nx, nx+1, -nx-1], format='csc')
    return A

def get_b(nx,ny,T_up = 20,T_low = 60,T_left = 60,T_right = 60):
    b = np.zeros((ny,nx))
    b[0] = T_up
    b[-1] = T_low
    b[:,0] = T_left
    b[:,-1] = T_right
    b = b.reshape(-1,) * -4
    return b
 

def solve(nx,ny,data_path1,data_path2,data_path3,level,datasize):
    alpha = sio.loadmat(data_path1)['b']
    beta = sio.loadmat(data_path2)['b']
    gamma = sio.loadmat(data_path3)['b']
    b = get_b(nx,ny)
    T = np.empty((len(alpha),nx*ny))
    for i in range(len(alpha)):
        print(i)
        A = get_A(nx,ny,alpha[i],beta[i],gamma[i])
        t = spsolve(A, b)
        # print(t)
        T[i] = t
    sio.savemat(f'./dataset/{datasize}/T/Dirichlet/T'+str(level)+'.mat',{'b':T})

if __name__ == '__main__':
    datasize = 200
    for i in range(4):
        print(i)
        nx = 16 * ( 2 ** i )
        ny = 4 * ( 2 ** i )
        level = 4 - i
        # name = 500 * i
        solve(nx,ny,f'./dataset/{datasize}/alpha/alpha'+str(level),f'./dataset/{datasize}/beta/beta'+str(level),f'./dataset/{datasize}/gamma/gamma'+str(level),level,datasize)
