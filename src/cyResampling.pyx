import numpy as np
cimport numpy as np
from numpy.random import random_sample

def resampling(np.ndarray[np.float64_t, ndim=1] w, scheme='multinomial'):
    cdef int N = w.shape[0]
    cdef int j = 0
    cdef int R = N
    cdef np.ndarray[np.float64_t, ndim=1] U = np.zeros(N, dtype=np.float64)
    cdef np.ndarray[np.int_t, ndim=1] ind = np.arange(N, dtype=np.int)
    cdef np.ndarray[np.float64_t, ndim=1] bins = np.zeros(N, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] wBar = np.zeros(N, dtype=np.float64)
    cdef np.ndarray[np.int_t, ndim=1] Ni = np.arange(N, dtype=np.int)
    
    if scheme == 'multinomial':
        w /= np.sum(w)
        bins = np.cumsum(w)
        ind = np.arange(N)[np.digitize(random_sample(N), bins)]
    elif scheme == 'residual':
        R = np.sum( np.floor(N * w).astype(int) )
        if R == N:
            ind = np.arange(N, dtype=np.int)
        else:
            wBar = (N * w - np.floor(N * w)) / (N-R)
            Ni = (np.floor(N*w) + np.random.multinomial(N-R, wBar)).astype(int)
            for i in range(N):
                ind[j:j+Ni[i]] = i
                j += Ni[i]
    elif scheme == 'stratified':
        U = (np.arange(N)+np.random.rand(N))/N
        bins = np.cumsum(w)
        ind = ind[np.digitize(U, bins)]
    elif scheme == 'systematic':
        U = (np.arange(N) + np.random.rand(1))/N
        bins = np.cumsum(w)
        ind = ind[np.digitize(U, bins)]
    return ind 
