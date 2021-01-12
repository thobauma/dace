# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
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


# @dace.program(dace.float64, dace.float64[N], dace.float64[N])
# def axpyMulti2(A, X, Y):
#     for k in dace.map[0:2]:
#         @dace.map(_[0:N])
#         def multiplication(i):
#             in_A << A
#             in_X << X[i]
#             in_Y << Y[i]
#             out >> Y[i]

#             out = in_A * in_X + in_Y + k



# def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
#     """ Finds the first map entry node by the given parameter name. """
#     return next(n for n, _ in sdfg.all_nodes_recursive()
#                 if isinstance(n, dace.nodes.MapEntry) and pname in n.params)

def test_gpu_multi():
    size = 256

    np.random.seed(0)
    A = np.random.rand()
    X = np.random.rand(size)
    Y = np.random.rand(size)
    Z = Y
    
    sdfg: dace.SDFG = axpyMultiGPU.to_sdfg()
    sdfg.apply_strict_transformations()
    # me = find_map_by_param(sdfg, 'i')
    #GPUMultiTransformMap.apply_to(sdfg, map_entry=me)
    # GPUTransformMap.apply_to(sdfg, _map_entry=me)
    sdfg.apply_transformations(GPUMultiTransformMap)
    
    # sdfg.view()
    sdfg.compile()
    # sdfg(A=A, X=X, Y=Y, N=size)

    # assert np.allclose(Y, A*X + Z)
    # print('PASS')


if __name__ == "__main__":
    test_gpu_multi()
