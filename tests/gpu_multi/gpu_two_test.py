import dace
import numpy as np
# import pytest
from dace.sdfg.sdfg import SDFG
from dace.transformation.dataflow import GPUTransformMap
from dace.sdfg import nodes
from dace.data import Scalar

N = dace.symbol('N')


@dace.program(dace.float64, dace.float64[N], dace.float64[N])
def axpy2GPU(A, X, Y):
    X1 = X[:N / 2]
    Y1 = Y[:N / 2]

    X2 = X[N / 2:]
    Y2 = Y[N / 2:]

    @dace.map(_[0:N / 2])
    def multiplication(i):
        in_A1 << A
        in_X1 << X1[i]
        in_Y1 << Y1[i]
        out1 >> Y[i]

        out1 = in_A1 * in_X1 + in_Y1

    @dace.map(_[0:N / 2])
    def multiplication(j):
        in_A2 << A
        in_X2 << X2[j]
        in_Y2 << Y2[j]
        out2 >> Y[j + N / 2]

        out2 = in_A2 * in_X2 + in_Y2


def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


def add_gpu_location(sdfg: dace.SDFG, mapEntry, gpu):
    graph = sdfg.nodes()[sdfg.sdfg_id]
    mapEntry.location = {'gpu': gpu}
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
        and not isinstance(e.src.desc(sdfg), Scalar)
    ]
    for e in entry_edges:
        data_node = e.src
        data_node.desc(sdfg).location = {'gpu': gpu}


# @pytest.mark.gpu
def test_two_gpus():
    size = 256

    np.random.seed(0)
    A = np.random.rand()
    X = np.random.rand(size)
    Y = np.random.rand(size)
    Z = np.copy(Y)

    sdfg: dace.SDFG = axpy2GPU.to_sdfg(strict=True)

    map1 = find_map_by_param(sdfg, 'i')
    map2 = find_map_by_param(sdfg, 'j')
    GPUTransformMap.apply_to(sdfg, _map_entry=map1)
    GPUTransformMap.apply_to(sdfg, _map_entry=map2)
    add_gpu_location(sdfg, map1, 0)
    add_gpu_location(sdfg, map2, 1)

    sdfg(A=A, X=X, Y=Y, N=size)

    assert np.allclose(Y, A * X + Z)
    print('PASS')


if __name__ == "__main__":
    test_two_gpus()
