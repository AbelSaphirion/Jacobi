import numpy as np
import numba as mb
from numba.experimental import jitclass
import argparse

@jitclass([('tgn', mb.types.float64),('cos', mb.types.float64),('sin', mb.types.float64)])
class Phi:
    def __init__(self, tgn : np.float64, cos : np.float64, sin : np.float64) -> None:
        self.tgn = tgn
        self.cos = cos
        self.sin = sin

@jitclass([('e', mb.types.float64), ('a', mb.types.float64[:,:]), ('height', mb.types.int64), ('s', mb.types.float64), 
           ('v', mb.types.float64[:,:]), ('tempa', mb.types.float64[:,:]), ('tempv', mb.types.float64[:,:])])
class Jacobi:
    def __init__(self, a : np.array([[]]), e : np.float64):
        assert len(a.shape) == 2, "Matrix must have 2 dimensions"
        assert a.shape[0] == a.shape[1], "Matrix must be simmetrical"
        assert np.allclose(a, a.T), "Matrix must be simmetrical"
        self.e = e
        self.a = a
        self.height = len(a)
        self.s = np.sum(a ** 2) - np.sum(np.diag(a) ** 2)
        self.v = np.ones(a.shape)
        self.update_temparr()

    def calc_phi(self, p: np.int64, q: np.int64) -> Phi:
        assert p != q, "p equals q"
        c = (self.tempa[q, q] - self.tempa[p, p]) / (2 * self.tempa[p, q])
        if c >= 0:
            tgphi = 1 / (c + np.sqrt(c ** 2 + 1))
        else:
            tgphi = 1 / (c - np.sqrt(c ** 2 + 1))
        cosphi = 1 / (np.sqrt(1 + tgphi ** 2))
        sinphi = tgphi * cosphi
        return Phi(tgphi, cosphi, sinphi)
    
    def update_temparr(self):
        self.tempa = np.copy(self.a)
        self.tempv = np.copy(self.v)

    def zero_element(self, p: np.int64, q: np.int64):
        phi = self.calc_phi(p, q)
        self.a[p, q] = self.a[q, p] = 0
        self.a[p, p] = self.tempa[p, p] - self.tempa[p, q] * phi.tgn
        self.a[q, q] = self.tempa[q, q] + self.tempa[p, q] * phi.tgn
        for i in mb.prange(self.height):
            if i == p or i == q:
                continue
            self.a[i, p] = self.tempa[i, p] - phi.sin * (self.tempa[i, q] + phi.sin / (1 + phi.cos) * self.tempa[i, p])
            self.a[p, i] = self.tempa[i, p] - phi.sin * (self.tempa[i, q] + phi.sin / (1 + phi.cos) * self.tempa[i, p])
            self.v[i, p] = self.tempv[i, p] * phi.cos - self.tempv[i, q] * phi.sin
            self.a[i, q] = self.tempa[i, q] + phi.sin * (self.tempa[i, p] - phi.sin / (1 + phi.cos) * self.tempa[i, q])
            self.a[q, i] = self.tempa[i, q] + phi.sin * (self.tempa[i, p] - phi.sin / (1 + phi.cos) * self.tempa[i, q])
            self.v[i, q] = self.tempv[i, p] * phi.sin + self.tempv[i, q] * phi.cos

@mb.njit(parallel=True)
def iterate(j : Jacobi, p : np.int64, q : np.int64):
    j.update_temparr()
    for i in range(0, p - q + 1):
        if (p - i != q + i) and (not np.isclose(j.tempa[p - i, q + i], 0)):
            j.zero_element(p - i, q + i)
            j.update_temparr()
    j.s = np.sum(j.a ** 2) - np.sum(np.diag(j.a) ** 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Jacobi calc.')
    parser.add_argument('filepath', type=str,
                    help='Path to a csv file.')
    parser.add_argument('-e', '--epsilon', type=np.float64, default=0.0001,
                    help='Calculation error.')
    parser.add_argument('-t', '--threads', type=int, default=1,
                    help='Number of parallel threads.')
    args = parser.parse_args()
    mb.set_num_threads(args.threads)
    a = np.genfromtxt(args.filepath, delimiter=',')
    assert a.size > 1, "Can't calc matrix with size 0 or 1"
    j = Jacobi(a, args.epsilon)
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
        print(f"\t\tIteration №{it}\n\t\tS = {j.s}")
        print(j.a)
        it += 1
