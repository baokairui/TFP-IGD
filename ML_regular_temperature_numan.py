import numpy as np
import scipy
import scipy.io as sio
from scipy.sparse.linalg import spsolve


# 四阶中心差分
def get_A(nx,ny,m1,m2,m):
    m1 = m1[1:-1,1:-1]
    m2 = m2[1:-1,1:-1]
    m1 = np.pad(m1,((1,1),(1,1)),'constant',constant_values=1).reshape(-1,)
    m2 = np.pad(m2,((1,1),(1,1)),'constant',constant_values=1).reshape(-1,)

    d01 = - np.ones(nx) * 4
    d02 = - np.ones(nx) * 2
    d02 = np.tile(d02, ny-2)
    d02 = d02 * (m1[nx:-nx] + m2[nx:-nx])
    d0 = np.concatenate((d01,d02,d01),axis=0)

    d11 = 1 + m[0]
    d11[0] = 0
    d11[-1] = 0
    d12 = np.ones(nx)
    d12[0] = 0
    d12[-1] = 0
    d12 = np.tile(d12, ny-2)
    d12 = d12 * m1[nx:-nx]
    d13 = 1 - m[-1][:-1]
    d13[0] = 0
    d1 = np.concatenate((d11,d12,d13),axis=0)
    
    d11_ = 1 - m[0][1:]
    d11_[-1] = 0
    d12_ = np.ones(nx)
    d12_[0] = 0
    d12_[-1] = 0
    d12_ = np.tile(d12_, ny-2)
    d12_ = d12_ * m1[nx:-nx]
    d13_ = 1 + m[-1]
    d13_[0] = 0
    d13_[-1] = 0
    d1_ = np.concatenate((d11_,d12_,d13_),axis=0)

    d21 = np.ones(nx) * 2
    d21[0] = 0
    d21[-1] = 0
    d22 = np.ones(nx)
    d22[0] = 0
    d22[-1] = 0
    d22 = np.tile(d22, ny-2)
    d22 = d22 * m2[nx:-nx]
    d2 = np.concatenate((d21,d22),axis=0)

    d21_ = np.ones(nx)
    d21_[0] = 0
    d21_[-1] = 0
    d21_ = np.tile(d21_, ny-2)
    d21_ = d21_ * m2[nx:-nx]
    d22_ = np.ones(nx) * 2
    d22_[0] = 0
    d22_[-1] = 0
    d2_ = np.concatenate((d21_,d22_),axis=0)

    A = scipy.sparse.diags([d0, d1, d1_, d2, d2_], [0, 1, -1, nx, -nx], format='csc')
    return A

def get_b(nx,ny,n,T_left = 20,T_right = 20):
    b = np.zeros((ny,nx))
    b[0] = - n[0]
    b[-1] = - n[-1]
    b[:,0] = - T_left * 4
    b[:,-1] = - T_right * 4
    b = b.reshape(-1,)
    return b

def solve(nx,ny,data_path1,data_path2,level):
    m1 = sio.loadmat(data_path1)['b']
    m2 = sio.loadmat(data_path2)['b']
    b = get_b(nx,ny)
    T = np.empty((len(m1),nx*ny))
    for i in range(len(m1)):
        A = get_A(nx,ny,m2[i],m1[i])
        t = spsolve(A, b)
        T[i] = t
    sio.savemat('T'+str(level)+'.mat',{'b':T})

if __name__ == '__main__':
    for i in range(1,5):
        print(i)
        nx = 16 * ( 2 ** i )
        ny = 4 * ( 2 ** i )
        level = 5 - i
        solve(nx,ny,'./dataset/m1/m1'+str(level),'./dataset/m2/m2'+str(level),level)
