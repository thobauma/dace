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

@dace.program(dace.float64, dace.float64[N], dace.float64[N])
def axpy2(A, X, Y):
    @dace.map(_[0:N])
    def multiplication(j):
        in_A << A
        in_X << X[j]
        in_Y << Y[j]
        out >> Y[j]

        out = in_A * in_X + in_Y

@dace.program(dace.float64, dace.float64, dace.float64, dace.float64, dace.float64[N], dace.float64[N], dace.float64[N], dace.float64[N], dace.float64[N], dace.float64[N])
def compound(a,b,c,alpha,A,B,C,D,X,Z):
    axpy(a,A,X)
    axpy(c,A,Z)
    axpy2(b,B,D)
    # axpy(b,A,X)
    axpy(alpha,C,X)
    # axpy(alpha,B,X)

def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)

def add_gpu_location(sdfg: dace.SDFG, mapEntry, gpu):
    mapEntry.location['gpu']=gpu
    graph = sdfg.nodes()[sdfg.sdfg_id]
    exit_edges = [
            e for e in graph.out_edges(mapEntry)
            if isinstance(e.dst, dace.nodes.Tasklet)
        ]
    for e in exit_edges:
        tasklet = e.dst
        tasklet.location = {'gpu': gpu}
    entry_edges = [
            e for e in graph.in_edges(mapEntry)
            if isinstance(e.src, dace.nodes.AccessNode)
        ]
    for e in entry_edges:
        data_node = e.src
        data_node.desc(sdfg).location = {'gpu': gpu}

# @pytest.mark.gpu
def test_compound():
    size = 256
    N.set(size)
    np.random.seed(0)
    a = np.random.rand()
    b = np.random.rand()
    c = np.random.rand()
    alpha = np.random.rand()
    A = np.random.rand(N.get()).astype(np.float64)
    B = np.random.rand(N.get()).astype(np.float64)
    C = np.random.rand(N.get()).astype(np.float64)
    D = np.random.rand(N.get()).astype(np.float64)
    X = np.random.rand(N.get()).astype(np.float64)
    Z = np.random.rand(N.get()).astype(np.float64)
    Zcopy = np.copy(Z)
    Xcopy = np.copy(X)
    Dcopy = np.copy(D)

    sdfg: dace.SDFG = compound.to_sdfg()
    sdfg.apply_strict_transformations()
    sdfg.apply_transformations(GPUTransformSDFG)
    
    map_entry = find_map_by_param(sdfg, 'j')

    add_gpu_location(sdfg, map_entry, 1)

    # sdfg.view()
    # sdfg.compile()
    sdfg(a=a,b=b,c=c,alpha=alpha,A=A,B=B,C=C,D=D,X=X,Z=Z)

    assert np.allclose(D, b * B + Dcopy)
    print('PASS 1/3')
    assert np.allclose(Z, c * A + Zcopy)
    print('PASS 2/3')
    assert np.allclose(X, alpha * C+(a*A+Xcopy))
    print('PASS 3/3')


if __name__ == "__main__":
    test_compound()
