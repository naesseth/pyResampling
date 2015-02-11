cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def resampling(np.ndarray w, ):
	cdef int N = w.shape[0]
	
