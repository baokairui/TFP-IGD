from unicodedata import name
import numpy as np
import scipy
import scipy.io as sio
from scipy.sparse.linalg import spsolve


## 四阶中心差分
# def get_A(nx,ny,m):
#     m0 = m[1:-1,1:-1]
#     m0 = np.pad(m0, ((1,1),(1,1)), 'constant', constant_values=2).reshape(-1,)
#     d01 = np.ones(nx)
#     d02 = np.ones(nx) * -4
#     d02[0] = 1
#     d02[-1] = 1
#     d02 = np.tile(d02, ny-2)
#     d0 = np.concatenate((d01,d02,d01),axis=0)
#     d0 = 0.5 * m0 * d0 - 2

#     d11_ = np.zeros(nx-1)
#     d12_ = np.ones(nx)
#     d12_[0] = 0
#     d12_[-1] = 0
#     d12_ = np.tile(d12_, ny-2)
#     d13_ = np.zeros(nx)
#     d1_ = np.concatenate((d11_,d12_,d13_),axis=0)
#     d1 = np.concatenate((d13_,d12_,d11_),axis=0)
    
#     m1 = m.reshape(-1,)
#     d21 = np.ones(nx)
#     d21[0] = 0
#     d21[-1] = 0
#     d21 = np.tile(d21, ny-2)
#     d22 = np.zeros(nx)
#     d2 = np.concatenate((d22,d21),axis=0)
#     d2 = m1[nx:] * d2
#     d2_ = np.concatenate((d21,d22),axis=0)
#     d2_ = m1[:-nx] * d2_
#     A = scipy.sparse.diags([d0, d1, d1_, d2, d2_], [0, 1, -1, nx, -nx], format='csc')
#     return A

# def get_b(nx,ny,T_up = 60,T_low = 22,T_left = 60,T_right = 60):
#     b = np.zeros((ny,nx))
#     b[0] = T_up
#     b[-1] = T_low
#     b[:,0] = T_left
#     b[:,-1] = T_right
#     b = -b.reshape(-1,)
#     return b

# def solve(nx,ny,data_path,level):
#     m = sio.loadmat(data_path)['b']
#     b = get_b(nx,ny)
#     T = np.empty((len(m),nx*ny))
#     for i,item in enumerate(m):
#         A = get_A(nx,ny,item)
#         t = spsolve(A, b)
#         T[i] = t
#     sio.savemat('T'+str(level)+'.mat',{'b':T})

# 边界六阶 中心四阶差分
def get_A(nx,ny,m1,m2):
    # m1,m2与实际相反
    m1 = m1[1:-1,1:-1]
    m2 = m2[1:-1,1:-1]
    m1 = np.pad(m1,((1,1),(1,1)),'constant',constant_values=1).reshape(-1,)
    m2 = np.pad(m2,((1,1),(1,1)),'constant',constant_values=1).reshape(-1,)

    d01 = np.ones(nx)
    d02 = np.ones(nx) * -2
    d02[0] = 1
    d02[-1] = 1
    d03 = np.ones(nx) * -51 / 2
    d03[0] = 1
    d03[-1] = 1
    d03[1] = -2
    d03[-2] = -2
    d03 = np.tile(d03, ny-4)
    d0 = np.concatenate((d01,d02,d03,d02,d01),axis=0)
    d0 = d0 * (m1 + m2)

    d11 = np.zeros(nx)
    d12 = np.ones(nx)
    d12[0] = 0
    d12[-1] = 0
    d13 = np.ones(nx) * 12
    d13[0] = 0
    d13[-1] = 0
    d13[1] = 1
    d13[-2] = 1
    d13 = np.tile(d13, ny-4)
    d14 = np.zeros(nx-1)
    d1 = np.concatenate((d11,d12,d13,d12,d14),axis=0)
    d1 = d1 * m2[:-1]

    d11_ = np.zeros(nx-1)
    d12_ = np.ones(nx)
    d12_[0] = 0
    d12_[-1] = 0
    d13_ = np.ones(nx) * 12
    d13_[0] = 0
    d13_[-1] = 0
    d13_[1] = 1
    d13_[-2] = 1
    d13_ = np.tile(d13_, ny-4)
    d14_ = np.zeros(nx)
    d1_ = np.concatenate((d11_,d12_,d13_,d12_,d14_),axis=0)
    d1_ = d1_ * m2[1:]

    d21 = np.zeros(nx)
    d22 = np.ones(nx)
    d22[0] = 0
    d22[-1] = 0
    d23 = np.ones(nx) * 12
    d23[0] = 0
    d23[-1] = 0
    d23[1] = 1
    d23[-2] = 1
    d23 = np.tile(d23, ny-4)
    d2 = np.concatenate((d21,d22,d23,d22),axis=0)
    d2 = d2 * m1[:-nx]

    d21_ = np.ones(nx)
    d21_[0] = 0
    d21_[-1] = 0
    d22_ = np.ones(nx) * 12
    d22_[0] = 0
    d22_[-1] = 0
    d22_[1] = 1
    d22_[-2] = 1
    d22_ = np.tile(d22_, ny-4)
    d23_ = np.zeros(nx)
    d2_ = np.concatenate((d21_,d22_,d21_,d23_),axis=0)
    d2_ = d2_ * m1[nx:]

    d31 = np.zeros(nx*2)
    d32 = np.ones(nx) * 0.75
    d32[0] = 0
    d32[-1] = 0
    d32[1] = 0
    d32[-2] = 0
    d32 = np.tile(d32, ny-4)
    d33 = np.zeros(nx*2-2)
    d3 = np.concatenate((d31,d32,d33),axis=0)
    d3 = d3 * m2[:-2]

    d31_ = np.zeros(nx*2-2)
    d32_ = np.ones(nx) * 0.75
    d32_[0] = 0
    d32_[-1] = 0
    d32_[1] = 0
    d32_[-2] = 0
    d32_ = np.tile(d32_, ny-4)
    d33_ = np.zeros(nx*2)
    d3_ = np.concatenate((d31_,d32_,d33_),axis=0)
    d3_ = d3_ * m2[2:]

    d41 = np.zeros(nx*2)
    d42 = np.ones(nx) * 0.75
    d42[0] = 0
    d42[-1] = 0
    d42[1] = 0
    d42[-2] = 0
    d42 = np.tile(d42, ny-4)
    d4 = np.concatenate((d41,d42),axis=0)
    d4 = d4 * m1[:-2*nx]

    d41_ = np.ones(nx) * 0.75
    d41_[0] = 0
    d41_[-1] = 0
    d41_[1] = 0
    d41_[-2] = 0
    d41_ = np.tile(d41_, ny-4)
    d42_ = np.zeros(nx*2)
    d4_ = np.concatenate((d41_,d42_),axis=0)
    d4_ = d4_ * m1[2*nx:]
    A = scipy.sparse.diags([d0, d1, d1_, d2, d2_, d3, d3_, d4, d4_], [0, 1, -1, nx, -nx, 2, -2, 2*nx, -2*nx], format='csc')
    return A

def get_b(nx,ny,T_up = 60,T_low = 22,T_left = 60,T_right = 60):
    b = np.zeros((ny,nx))
    b[0] = T_up
    b[-1] = T_low
    b[:,0] = T_left
    b[:,-1] = T_right
    b = b.reshape(-1,) * 2
    return b

def solve(nx,ny,data_path1,data_path2,level):
    m1 = sio.loadmat(data_path1)['b']
    m2 = sio.loadmat(data_path2)['b']
    b = get_b(nx,ny)
    T = np.empty((len(m1),nx*ny))
    for i in range(len(m1)):
        A = get_A(nx,ny,m2[i],m1[i])
        t = spsolve(A, b)
        # print(t)
        T[i] = t
    sio.savemat('T'+str(level)+'.mat',{'b':T})

if __name__ == '__main__':
    for i in range(1,5):
        print(i)
        nx = 16 * ( 2 ** i )
        ny = 4 * ( 2 ** i )
        level = 5 - i
        name = 500 * i
        solve(nx,ny,'./dataset/2000/gamma/gamma'+str(level),'./dataset/2000/alpha/alpha'+str(level),level)
