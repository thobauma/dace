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
    axpy(c,A,Z)
    # axpy(b,B,A)
    # axpy(b,A,X)
    axpy(alpha,C,X)
    # axpy(alpha,B,X)

def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


def add_gpu_location(sdfg: dace.SDFG, mapEntry, gpu):
    graph = sdfg.nodes()[sdfg.sdfg_id]
    exit_edges = [
            e for e in graph.out_edges(mapEntry)
            if isinstance(e.dst, nodes.Tasklet)
        ]
    for e in exit_edges:
        tasklet = e.dst
        tasklet.location = {'gpu': gpu}
    entry_edges = [
            e for e in graph.in_edges(mapEntry)
            if isinstance(e.src, nodes.AccessNode)
        ]
    for e in entry_edges:
        data_node = e.src
        data_node.desc(sdfg).location = {'gpu': gpu}

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


    sdfg.view()
    sdfg.compile()
    # sdfg(A=A, X=X, Y=Y, N=size)

    # assert np.allclose(Y, A * X + Z)
    # print('PASS')


if __name__ == "__main__":
    test_compound()
