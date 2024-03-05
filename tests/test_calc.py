import pytest
from jacobi import Jacobi
import numpy as np

def iterate(j : Jacobi, p : np.int64, q : np.int64):
    j.update_temparr()
    for i in range(0, p - q + 1):
        if (p - i != q + i) and (not np.isclose(j.tempa[p - i, q + i], 0)):
            j.zero_element(p - i, q + i)
            j.update_temparr()
    j.s = np.sum(j.a ** 2) - np.sum(np.diag(j.a) ** 2)

def loop(j):
    p = q = np.int64(0)
    it = 1
    while j.s >= j.e:
        if(p == q):
            if(p < j.height - 1):
                p += 1
            else:
                q += 1
        if(q >= j.height):
            p = q = np.int64(0)
            continue
        iterate(j, p, q)
        if(p < j.height - 1):
            p += 1
        else:
            q += 1
        it += 1

def test_calc3x3():
    a = np.array([[3.33,-7.15,8.00],
                  [-7.15,4.17,-0.07],
                  [8.00,-0.07,-5.12]])
    j = Jacobi(a, 10e-10)
    loop(j)
    assert np.allclose(np.sort(np.diag(j.a)), np.array([-10.94636057850104, 0.4854684575394801, 12.84089212096156]))

def test_calc5x5():
    a = np.array([[ 3.79 , 1.09 , 9.72 , 0.35 ,-9.97],
                  [1.09 , 3.61 ,-6.51 ,-4.50 , 5.20],
                  [9.72 ,-6.51 , 9.07 ,-4.13 ,-3.75],
                  [0.35 ,-4.50 ,-4.13 ,-3.45 ,-2.35],
                  [-9.97 , 5.20 ,-3.75 ,-2.35 , 1.08]])
    j = Jacobi(a, 10e-10)
    loop(j)
    assert np.allclose(np.sort(np.diag(j.a)), np.array([-11.82631846573668, -7.124446471100766, 3.895755904998749, 6.363794509693024, 22.791214522145676]))

def test_calc10x10():
    a = np.array([[ 6.74 , 6.94 ,-1.12 , 6.31 , 7.49 ,-2.67 , 7.64 , 0.17 ,-3.48 , 9.82],
                  [ 6.94 ,-4.79 , 6.40 ,-6.88 , 9.81 , 0.82 , 9.58 , 9.61 ,-6.88 ,-5.17],
                  [-1.12 , 6.40 ,-8.91 ,-7.15 , 4.13 , 7.81 , 4.12 , 3.17 ,-4.78 , 9.87],
                  [ 6.31 ,-6.88 ,-7.15 , 2.25 , 7.91 ,-5.84 , 2.72 , 4.64 , 4.63 ,-1.86],
                  [ 7.49 , 9.81 , 4.13 , 7.91 , 7.01 ,-9.05 ,-5.94 ,-3.81 , 9.55 ,-6.44],
                  [-2.67 , 0.82 , 7.81 ,-5.84 ,-9.05 , 9.39 ,-0.31 , 0.17 ,-3.06 , 9.57],
                  [ 7.64 , 9.58 , 4.12 , 2.72 ,-5.94 ,-0.31 ,-6.10 ,-5.11 , 7.02 , 6.80],
                  [ 0.17 , 9.61 , 3.17 , 4.64 ,-3.81 , 0.17 ,-5.11 , 6.47 , 3.42 ,-9.30],
                  [-3.48 ,-6.88 ,-4.78 , 4.63 , 9.55 ,-3.06 , 7.02 , 3.42 , 7.83 , 9.23],
                  [ 9.82 ,-5.17 , 9.87 ,-1.86 ,-6.44 , 9.57 , 6.80 ,-9.30 , 9.23 ,-4.01]])
    j = Jacobi(a, 10e-10)
    loop(j)
    assert np.allclose(np.sort(np.diag(j.a)), np.sort(np.linalg.eig(a)[0]))
