# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
# import pytest
from dace.transformation.interstate import GPUTransformSDFG

N = dace.symbol('N')


@dace.program(dace.float64, dace.float64[N], dace.float64[N])
def axpy(A, X, Y):
    @dace.map(_[0:N])
    def multiplication(i):
        in_A << A
        in_X << X[i]
        in_Y << Y[i]
        out >> Y[i]

        out = in_A * in_X + in_Y

@dace.program(dace.float64, dace.float64, dace.float64, dace.float64, dace.float64[N], dace.float64[N], dace.float64[N], dace.float64[N], dace.float64[N])
def compound(a,b,c,alpha,A,B,C,X,Z):
    axpy(a,A,X)
    axpy(c,C,Z)
    axpy(b,B,A)
    axpy(b,A,X)
    axpy(alpha,Z,X)
    axpy(alpha,B,X)

# @pytest.mark.gpu
def test_compound():
    size = 256

    np.random.seed(0)
    A = np.random.rand()
    X = np.random.rand(size)
    Y = np.random.rand(size)
    Z = np.copy(Y)

    sdfg: dace.SDFG = compound.to_sdfg()
    sdfg.apply_strict_transformations()
    sdfg.apply_transformations(GPUTransformSDFG)

    # sdfg.view()
    sdfg.compile()
    # sdfg(A=A, X=X, Y=Y, N=size)

    # assert np.allclose(Y, A * X + Z)
    # print('PASS')


if __name__ == "__main__":
    test_compound()
