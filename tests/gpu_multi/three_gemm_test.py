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
def three_matmul_strict(A: dtype[M, K], B: dtype[K, N], C: dtype[N, L],
                        D: dtype[L, O]):
    M1 = matmul_lib(A, B)
    M2 = matmul_lib(C, D)

    return matmul_lib(M1, M2)


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

def test_gemm():
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


def test_three_gemm_strict():
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
    dace.libraries.blas.default_implementation = 'cuBLAS'

    sdfg: dace.SDFG = three_matmul_strict.to_sdfg()
    sdfg.expand_library_nodes()
    print('expanded')
    sdfg.apply_strict_transformations()
    print('strict_transformations')
    graph = sdfg.nodes()[0]

    tasklets = find_node_type(sdfg, graph, nodes.Tasklet)
    nsdfg = find_node_type(sdfg, graph, nodes.NestedSDFG)[0].sdfg

    set_gpu_location(sdfg, graph, tasklets[0], 1)
    set_gpu_location(sdfg, graph, tasklets[1], 0)
    sdfg.arrays['M1'].location = {'gpu': 1}
    sdfg.arrays['M1'].storage = StorageType.GPU_Global

    sdfg.arrays['M2'].location = {'gpu': 1}
    sdfg.arrays['M2'].storage = StorageType.GPU_Global
    ngraph = nsdfg.nodes()[0]
    ntasklet = find_node_type(nsdfg, ngraph, nodes.Tasklet)[0]
    set_gpu_location(sdfg, ngraph, ntasklet, 1)
    nsdfg.arrays['__tmp5'].location = {'gpu': 1}
    nsdfg.arrays['__tmp5'].storage = StorageType.GPU_Global
    nsdfg.arrays['__tmp6'].location = {'gpu': 1}
    nsdfg.arrays['__tmp6'].storage = StorageType.GPU_Global
    sdfg.apply_strict_transformations()

    E = sdfg(A=A, B=B, C=C, D=D, M=m, K=k, N=n, L=l, O=o)
    assert np.allclose(E, (A @ B) @ (C @ D))
    print('PASS')


def test_three_gemm_not_strict():
    a = 32

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
    dace.libraries.blas.default_implementation = 'cuBLAS'

    sdfg: dace.SDFG = three_matmul.to_sdfg()
    sdfg.expand_library_nodes()
    m1sdfg = sdfg.sdfg_list[3]
    m2sdfg = sdfg.sdfg_list[4]
    m3sdfg = sdfg.sdfg_list[2]
    binsdfg = sdfg.sdfg_list[1]
    nsdfg_set_gpu_location(m1sdfg, 1)
    nsdfg_set_gpu_location(m3sdfg, 1)
    nsdfg_set_gpu_location(m2sdfg, 0)

    m1sdfg.arrays['_c'].location = {'gpu': 1}
    m1sdfg.arrays['_c'].storage = StorageType.GPU_Global
    m2sdfg.arrays['_c'].location = {'gpu': 0}
    m2sdfg.arrays['_c'].storage = StorageType.GPU_Global
    sdfg.arrays['M1'].location = {'gpu': 1}
    sdfg.arrays['M1'].storage = StorageType.GPU_Global
    sdfg.arrays['M2'].location = {'gpu': 1}
    sdfg.arrays['M2'].storage = StorageType.GPU_Global
    m3sdfg.arrays['_a'].location = {'gpu': 1}
    m3sdfg.arrays['_a'].storage = StorageType.GPU_Global
    m3sdfg.arrays['_b'].location = {'gpu': 1}
    m3sdfg.arrays['_b'].storage = StorageType.GPU_Global
    binsdfg.arrays['__tmp5'].location = {'gpu': 1}
    binsdfg.arrays['__tmp5'].storage = StorageType.GPU_Global
    binsdfg.arrays['__tmp6'].location = {'gpu': 1}
    binsdfg.arrays['__tmp6'].storage = StorageType.GPU_Global
    sdfg.apply_transformations_repeated(RedundantArray)
    sdfg.apply_transformations_repeated(RedundantSecondArray)

    # sdfg.view()
    # sdfg.compile()
    E = sdfg(A=A, B=B, C=C, D=D, M=m, K=k, N=n, L=l, O=o)
    assert np.allclose(E, (A @ B) @ (C @ D))
    print('PASS')


if __name__ == "__main__":
    test_three_gemm_not_strict()
    # test_three_gemm_strict()
    # test_gemm()
