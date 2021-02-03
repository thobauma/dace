import dace
import numpy as np
# import pytest
from dace.sdfg.sdfg import SDFG
from dace.transformation.dataflow import GPUTransformMap
from dace.sdfg import nodes
from dace.data import Scalar

N = dace.symbol('N')


@dace.program(dace.float64, dace.float64[N], dace.float64[N])
def axpySelGPU(A, X, Y):
    @dace.map(_[0:N])
    def multiplication(i):
        in_A << A
        in_X << X[i]
        in_Y << Y[i]
        out >> Y[i]

        out = in_A * in_X + in_Y


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
def test_select_gpu():
    size = 256

    np.random.seed(0)
    A = np.random.rand()
    X = np.random.rand(size)
    Y = np.random.rand(size)
    Z = np.copy(Y)

    sdfg: dace.SDFG = axpySelGPU.to_sdfg(strict=True)

    map_ = find_map_by_param(sdfg, 'i')
    GPUTransformMap.apply_to(sdfg, _map_entry=map_)
    add_gpu_location(sdfg, map_, 0)

    sdfg(A=A, X=X, Y=Y, N=size)

    assert np.allclose(Y, A * X + Z)
    print('PASS')


if __name__ == "__main__":
    test_select_gpu()
