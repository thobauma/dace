# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from typing import List

# Define symbolic sizes for arbitrary inputs
M = dace.symbol('M')
K = dace.symbol('K')
N = dace.symbol('N')
L = dace.symbol('L')
O = dace.symbol('O')

# Define data type to use
dtype = dace.float64
np_dtype = np.float64

#####################################################################
# Data-centric functions


# Map-Reduce version of matrix multiplication
@dace.program
def matmul(A: dtype[M, K], B: dtype[K, N], C: dtype[M, N]):
    tmp = np.ndarray([M, N, K], dtype=A.dtype)
    # Multiply every pair of values to a large 3D temporary array
    for i, j, k in dace.map[0:M, 0:N, 0:K]:
        with dace.tasklet:
            in_A << A[i, k]
            in_B << B[k, j]
            out >> tmp[i, j, k]

            out = in_A * in_B

    # Sum last dimension of temporary array to obtain resulting matrix
    dace.reduce(lambda a, b: a + b, tmp, C, axis=2, identity=0)


@dace.program
def matmul2(A: dtype[M, K], B: dtype[K, N], C: dtype[N, L], D: dtype[L, O],
            E: dtype[M, O]):
    tmpAB = np.ndarray([M, N], dtype=A.dtype)
    matmul(A, B, tmpAB)
    tmpCD = np.ndarray([N, O], dtype=A.dtype)
    matmul(C, D, tmpCD)

    matmul(tmpAB, tmpCD, E)


def test_streams():
    a = 16

    np.random.seed(0)
    m = a
    k = a
    n = a
    l = a
    o = a
    A = np.random.rand(m, k).astype(np_dtype)
    B = np.random.rand(k, n).astype(np_dtype)
    C = np.random.rand(n, l).astype(np_dtype)
    D = np.random.rand(l, o).astype(np_dtype)
    E = np.zeros((m, o), dtype=np_dtype)
    sdfg: dace.SDFG = matmul2.to_sdfg()
    sdfg.apply_strict_transformations()

    sdfg.view()

    # sdfg.compile()
    sdfg(A=A, B=B, C=C, D=D, E=E, M=m, K=k, N=n, L=l, O=o)
    assert np.allclose(E, (A @ B) @ (C @ D))
    print('PASS')


if __name__ == "__main__":
    test_streams()
