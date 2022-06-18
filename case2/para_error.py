import bezier
import numpy as np
import scipy.io as sio
from scipy.special import comb as nOk
from numpy import array, linalg, matrix
from ML_mesh_parameter import get_bezier,get_bound


def lsqfit(points,M):
    M_ = linalg.pinv(M)
    return M_ * points

para = sio.loadmat("./dataset/para-2000.mat")['b']
Mtk = lambda n, t, k: t**(k)*(1-t)**(n-k)*nOk(n,k)
bezierup = lambda ts: matrix([[Mtk(6,t,k) for k in range(7)] for t in ts])
bezierlow = lambda ts: matrix([[Mtk(8,t,k) for k in range(9)] for t in ts])
bezierleft = lambda ts: matrix([[Mtk(3,t,k) for k in range(4)] for t in ts])
V = array
ts = V(range(64), dtype='float')/63
Mup = bezierup(ts)
Mleft = bezierleft(ts)
Mlow = bezierlow(ts)
error = 0
for i, item in enumerate(para):
    print(i)
    upx,upy,lowx,lowy,leftx,lefty,rightx,righty = get_bound(64, 64, item)
    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8 = item

    up = matrix([[x1,y1],[x2,y2],[x3,y3],[0,y3],[-x3,y3],[-x2,y2],[-x1,y1]])
    left = matrix([[x5,0],[x5,y5],[x4,y4],[x1,y1]])
    low = matrix([[x5,0],[x6,y6],[x7,y7],[x8,y8],[0,y8],[-x8,y8],[-x7,y7],[-x6,y6],[-x5,0]])

    pointup = np.zeros((64,2))
    pointup[:,0] = upx
    pointup[:,1] = upy

    pointlow = np.zeros((64,2))
    pointlow[:,0] = lowx
    pointlow[:,1] = lowy

    pointleft = np.zeros((64,2))
    pointleft[:,0] = leftx
    pointleft[:,1] = lefty

    control_points_up = lsqfit(pointup, Mup)
    control_points_low = lsqfit(pointlow, Mlow)
    control_points_left = lsqfit(pointleft, Mleft)
    # temp = linalg.norm(control_points_up-up)
    temp = (linalg.norm(control_points_up-up) + linalg.norm(control_points_low-low) + linalg.norm(control_points_left-left)) / 3
    error = error + temp
    print(temp < 10e-10)
    # print(up)
    # print(control_points_up)
print(error/2000)