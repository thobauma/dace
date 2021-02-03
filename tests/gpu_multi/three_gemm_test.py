# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.sdfg import nodes, SDFG, SDFGState
from dace.data import Scalar

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
# Data-centric functions


# Map-Reduce version of matrix multiplication
@dace.program
def matmul_lib_transformed(A: dtype[M, K], B: dtype[K, N]):
    return A @ B


@dace.program
def matmul_lib(A: dtype[M, K], B: dtype[K, N]):
    return A @ B


@dace.program
def three_matmul(A: dtype[M, K], B: dtype[K, N], C: dtype[N, L], D: dtype[L,
                                                                          O]):
    M1 = matmul_lib(A, B)
    M2 = matmul_lib(C, D)

    matmul_lib(M1, M2)


@dace.program
def three_matmul_transformed(A: dtype[M, K], B: dtype[K, N], C: dtype[N, L],
                             D: dtype[L, O]):
    M1 = matmul_lib(A, B)
    M2 = matmul_lib(C, D)

    matmul_lib(M1, M2)


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

def find_codeNodes(sdfg:SDFG, graph:SDFGState):
    return [node for node in graph.nodes()
            if isinstance(node, nodes.CodeNode)]

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
    dace.libraries.blas.default_implementation = 'cuBLAS'
    # sdfg: dace.SFDG = matmul_lib.to_sdfg()
    # sdfg: dace.SFDG = matmul_lib_transformed.to_sdfg()
    # sdfg: dace.SDFG = three_matmul.to_sdfg()
    sdfg: dace.SDFG = three_matmul_transformed.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.apply_strict_transformations()

    graph = sdfg.nodes()[0]
    codeNodes = find_codeNodes(sdfg, graph)

    set_gpu_location(sdfg, graph, codeNodes[0], 0)
    set_gpu_location(sdfg, graph, codeNodes[1], 1)
    set_gpu_location(sdfg, graph, codeNodes[2], 0)

    # mmAB.expand(sdfg,graph)

    # sdfg.apply_strict_transformations()

    sdfg.view()

    sdfg.compile()
    # sdfg(A=A, B=B, C=C, D=D, E=E, M=m, K=k, N=n, L=l, O=o)
    # assert np.allclose(E, (A @ B) @ (C @ D))
    # print('PASS')


if __name__ == "__main__":
    test_streams()
