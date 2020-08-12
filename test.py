import numpy as np
from time import time
from numba import jit, njit, autojit, prange

@njit(parallel=True)
def parallel_square(A):
    sum = 0.0
    for i in prange(A.shape[0]):
        sum2 = 0.0
        for j in prange(A.shape[0]):
            sum2 += A[i]
        sum = np.add(sum,sum2)
    return sum

x = np.ones(1000000)
start1 = time()
print(parallel_square(x))
end1 = time()


print("Parallel Time: " + str(end1 - start1))
