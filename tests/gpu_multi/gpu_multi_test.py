# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
#import pytest
from dace.transformation.dataflow import GPUMultiTransformMap, GPUTransformMap

N = dace.symbol('N')


@dace.program(dace.float64, dace.float64[N], dace.float64[N])
def axpyMultiGPU(A, X, Y):
    @dace.map(_[0:N])
    def multiplication(i):
        in_A << A
        in_X << X[i]
        in_Y << Y[i]
        out >> Y[i]

        out = in_A * in_X + in_Y


#@pytest.mark.gpu
def test_gpu_multi():
    size = 256

    np.random.seed(0)
    A = np.random.rand()
    X = np.random.rand(size)
    Y = np.random.rand(size)
    Z = np.copy(Y)


    sdfg: dace.SDFG = axpyMultiGPU.to_sdfg()
    sdfg.apply_strict_transformations()
    sdfg.apply_transformations(GPUMultiTransformMap)
    #                           options={'number_of_gpus': 4})

    # sdfg.compile()
    sdfg(A=A, X=X, Y=Y, N=size)

    idx = zip(*np.where(~np.isclose(Y, A * X + Z, atol=0, rtol=1e-7)))
    for i in idx:
        print(i, Y[i], Z[i], A * X[i] + Z[i])
    assert np.allclose(Y, A * X + Z)
    print('PASS')


if __name__ == "__main__":
    test_gpu_multi()
