# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace

import numpy as np

def test_2gpus():
    size = 256

    np.random.seed(0)
    A = np.random.rand()
    X = np.random.rand(size)
    Y = np.random.rand(size)
    Z = Y
    sdfg = dace.SDFG.from_file('/Users/Thomas/Documents/eth/2020hs/thesis/dace/tests/delme/axpy2.sdfg')
    
    # sdfg.view()
    sdfg.compile()
    # sdfg(A=A, X=X, Y=Y, N=size)

    # assert np.allclose(Y, A*X + Z)
    # print('PASS')


if __name__ == "__main__":
    test_2gpus()
