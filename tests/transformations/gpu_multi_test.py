# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.transformation.dataflow import GPUMultiTransformMap

N = dace.symbol('N')


@dace.program(dace.float64, dace.float64[N], dace.float64[N])
def axpyMulti(A, X, Y):
    @dace.map(_[0:N])
    def multiplication(i):
        in_A << A
        in_X << X[i]
        in_Y << Y[i]
        out >> Y[i]

        out = in_A * in_X + in_Y


def test_gpu_multi():
    size = 256

    np.random.seed(0)
    A = np.random.rand()
    X = np.random.rand(size)
    Y = np.random.rand(size)
    Z = Y
    
    sdfg = axpyMulti.to_sdfg()
    sdfg.apply_strict_transformations()
    sdfg.apply_transformations(GPUMultiTransformMap)
    
    # sdfg.view()
    sdfg.compile()
    # sdfg(A=A, X=X, Y=Y, N=size)

    # assert np.allclose(Y, A*X + Z)
    # print('PASS')


if __name__ == "__main__":
    test_gpu_multi()
