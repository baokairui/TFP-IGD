import numpy as np
import scipy
import scipy.io as sio
from scipy.sparse.linalg import spsolve

def get_A(nx,ny,a,b,r,m):
    a = a[1:-1,1:-1]
    b = b[1:-1,1:-1]
    r = r[1:-1,1:-1]
    a = np.pad(a,((1,1),(1,1)),'constant',constant_values=1).reshape(-1,)
    b = np.pad(b,((1,1),(1,1)),'constant',constant_values=1).reshape(-1,)
    r = np.pad(r,((1,1),(1,1)),'constant',constant_values=1).reshape(-1,)

    d01 = - np.ones(nx) * 1.5
    d01[0] = -2
    d01[-1] = -2
    d02 = - np.ones(nx) * 2
    d02 = np.tile(d02, ny-2)
    d0 = np.concatenate((d01,d02,d01),axis=0)
    d0 =  d0 * (a + r) 

    d11 = np.ones(nx) 
    d11[0] = 0
    d11[-1] = 0
    d12 = np.ones(nx)
    d12[0] = 0
    d12[-1] = 0
    d12 = np.tile(d12, ny-2)
    d13 = np.ones(nx-1) 
    d13[0] = 0
    d1 = np.concatenate((-d11 * m[0],d12,d13 * m[-1][:-1]),axis=0)
    d1 = d1 * a[:-1]
    d13_ = np.ones(nx-1) 
    d13_[-1] = 0
    d1_ = np.concatenate((d13_ * m[0][1:],d12,-d11 * m[-1]),axis=0)
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

    d31 = np.ones(nx) * 4
    d31[0] = 0
    d31[-1] = 0
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

    d51 = np.ones(nx) * -1
    d51[0] = 0
    d51[-1] = 0
    d52 = np.zeros(nx)
    d52 = np.tile(d52, ny-3)
    d5 = np.concatenate((d51,d52),axis=0)
    d5 = d5 * r[:-2*nx]
    d5_ = np.concatenate((d52,d51),axis=0)
    d5_ = d5_ * r[2*nx:]

    A = scipy.sparse.diags([d0, d1, d1_, d2, d2_, d3, d3_, d4, d4_, d5, d5_], [0, 1, -1, nx-1, -nx+1, nx, - nx, nx+1, -nx-1, 2*nx, -2*nx], format='csc')
    return A

def get_b(nx,ny,q1,q2,n,T_left = 20,T_right = 20):
    b = np.zeros((ny,nx))
    b[0] = - n[0] * q1
    b[-1] = - n[-1] * q2
    b[:,0] = - T_left * 4
    b[:,-1] = - T_right * 4
    b = b.reshape(-1,)
    return b

def solve(nx,ny,data_path1,data_path2,data_path3,data_path4,level):
    h = 0.01
    alpha = sio.loadmat(data_path1)['b']
    beta = sio.loadmat(data_path2)['b']
    gamma = sio.loadmat(data_path3)['b']
    J = sio.loadmat(data_path4)['b']
    T = np.empty((len(alpha),nx*ny))
    for i in range(len(alpha)):
        print(i)
        n = 2 * h * J[i] / np.sqrt(gamma[i])
        A = get_A(nx,ny,alpha[i],beta[i],gamma[i],beta[i]/gamma[i])
        b = get_b(nx,ny,1,1,n)
        t = spsolve(A, b)
        # print(np.max(t))
        # print(t)
        T[i] = t
    sio.savemat('./dataset/1500/T/Neuman/T'+str(level)+'.mat',{'b':T})

if __name__ == '__main__':
    for i in range(1,5):
        print(i)
        nx = 4 * ( 2 ** i )
        ny = 4 * ( 2 ** i )
        level = 5 - i
        solve(nx,ny,'./dataset/1500/alpha/alpha'+str(level),'./dataset/1500/beta/beta'+str(level),'./dataset/1500/gamma/gamma'+str(level),'./dataset/1500/J/J'+str(level),level)
