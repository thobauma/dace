# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.dtypes import StorageType
from dace.sdfg import nodes, SDFG, SDFGState
from dace.data import Scalar
from dace.transformation.dataflow.redundant_array import RedundantArray, RedundantSecondArray

import dace.libraries.blas

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


@dace.program
def matmul_lib(A: dtype[M, K], B: dtype[K, N]):
    return A @ B


@dace.program
def three_matmul(A: dtype[M, K], B: dtype[K, N], C: dtype[N, L], D: dtype[L,
                                                                          O]):
    M1 = matmul_lib(A, B)
    M2 = matmul_lib(C, D)
    return matmul_lib(M1, M2)


@dace.program
def three_matmul_debug(A: dtype[M, K], B: dtype[K, N], C: dtype[N, L],
                       D: dtype[L, O]):
    M1 = matmul_lib(A, B)
    M2 = matmul_lib(C, D)
    rM1 = M1
    rM2 = M2
    return matmul_lib(M1, M2), rM1, rM2


def set_gpu_location(sdfg: SDFG, graph: SDFGState, codeNode: nodes.CodeNode,
                     gpu: int):
    codeNode.location = {'gpu': gpu}
    for e in graph.in_edges(codeNode):
        if isinstance(e.src, nodes.AccessNode) and not isinstance(
                e.src.desc(graph), Scalar):
            e.src.desc(graph).location = {'gpu': gpu}
    for e in graph.out_edges(codeNode):
        if isinstance(e.dst, nodes.AccessNode) and not isinstance(
                e.dst.desc(graph), Scalar):
            e.dst.desc(graph).location = {'gpu': gpu}


def find_node_type(sdfg: SDFG, graph: SDFGState, node_type):
    return [node for node in graph.nodes() if isinstance(node, node_type)]


def nsdfg_set_gpu_location(sdfg: SDFG, gpu: int):
    graph = sdfg.nodes()[0]
    tasklet = find_node_type(sdfg, graph, nodes.Tasklet)[0]
    set_gpu_location(sdfg, graph, tasklet, gpu)


def test_matmul():
    a = 16

    np.random.seed(0)
    m = a
    k = a
    n = a
    A = np.random.rand(m, k).astype(np_dtype)
    B = np.random.rand(k, n).astype(np_dtype)

    dace.libraries.blas.default_implementation = 'cuBLAS'

    sdfg: dace.SDFG = matmul_lib.to_sdfg()
    sdfg.expand_library_nodes()
    print('expanded')
    sdfg.apply_strict_transformations()

    C = sdfg(A=A, B=B)
    assert np.allclose(C, (A @ B))
    print('PASS')


def test_three_matmul():
    a = 32
    gpuHelper = 1
    gpuMain = 0

    np.random.seed(0)
    # m = a
    # k = a
    # n = a
    # l = a
    # o = a
    m = 1024
    k = 2000
    n = 3000
    l = 900
    o = 7777
    A = np.random.rand(m, k).astype(np_dtype)
    B = np.random.rand(k, n).astype(np_dtype)
    C = np.random.rand(n, l).astype(np_dtype)
    D = np.random.rand(l, o).astype(np_dtype)
    dace.libraries.blas.default_implementation = 'cuBLAS'

    sdfg: dace.SDFG = three_matmul.to_sdfg()
    sdfg.expand_library_nodes()
    mABsdfg = sdfg.sdfg_list[1]
    mCDsdfg = sdfg.sdfg_list[2]
    mEFsdfg = sdfg.sdfg_list[3]
    nsdfg_set_gpu_location(mABsdfg, gpuMain)
    nsdfg_set_gpu_location(mCDsdfg, gpuHelper)
    nsdfg_set_gpu_location(mEFsdfg, gpuMain)

    mABsdfg.arrays['_c'].location = {'gpu': gpuMain}
    mABsdfg.arrays['_c'].storage = StorageType.GPU_Global
    mCDsdfg.arrays['_c'].location = {'gpu': gpuMain}
    mCDsdfg.arrays['_c'].storage = StorageType.GPU_Global
    sdfg.arrays['M1'].location = {'gpu': gpuMain}
    sdfg.arrays['M1'].storage = StorageType.GPU_Global
    sdfg.arrays['M2'].location = {'gpu': gpuMain}
    sdfg.arrays['M2'].storage = StorageType.GPU_Global
    mEFsdfg.arrays['_a'].location = {'gpu': gpuMain}
    mEFsdfg.arrays['_a'].storage = StorageType.GPU_Global
    mEFsdfg.arrays['_b'].location = {'gpu': gpuMain}
    mEFsdfg.arrays['_b'].storage = StorageType.GPU_Global

    # sdfg.apply_transformations_repeated([RedundantArray, RedundantSecondArray])
    sdfg.apply_strict_transformations()

    # sdfg.view()
    # sdfg.compile()

    E = sdfg(A=A, B=B, C=C, D=D, M=m, K=k, N=n, L=l, O=o)
    res = (A @ B) @ (C @ D)
    idx = list(zip(*np.where(~np.isclose(E, res, atol=0, rtol=1e-7))))
    numErrors = len(idx)
    print("number of errors:", numErrors)
    if numErrors < 100:
        for i in idx:
            print(i, E[i], res[i])
    assert np.allclose(E, res)
    print('PASS')


if __name__ == "__main__":
    test_three_matmul()
    # test_matmul()
