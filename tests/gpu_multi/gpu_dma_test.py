# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.dtypes import StorageType
from dace.sdfg import nodes, SDFG, SDFGState, InterstateEdge
from dace.data import Scalar
from dace.memlet import Memlet
from dace.transformation.dataflow import GPUTransformMap
from dace.transformation.interstate import GPUTransformSDFG

N = dace.symbol('N')

# Define data type to use
dtype = dace.float64
np_dtype = np.float64

# @dace.program
# def dot(X: dtype[N], Y: dtype[N], out: dtype):
#     @dace.map
#     def product(i: _[0:N]):
#         a << X[i]
#         b << Y[i]
#         o >> out(1, lambda x, y: x + y)
#         o = a * b
    


# @dace.program
# def gpu_dma(X: dtype[N], Y: dtype[N], out: dtype):
#     @dace.map
#     def product(i: _[0:N]):
#         a << X[i]
#         b << Y[i]
#         o >> out(1, lambda x, y: x + y)
#         o = a * b

#     @dace.map
#     def mult(j: _[0:N]):
#         in_a << X[i]
#         in_b << out
#         out_x >> X[i]
#         out_x = in_a * in_b

@dace.program
def gpu_dma(X: dtype[N], alpha: dtype):
    return alpha*X


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
    

def gpu_dma_test():
    np.random.seed(0)
    n = 16
    X = np.random.rand(n).astype(np_dtype)
    alpha = np.random.rand()


    sdfg: dace.SDFG = gpu_dma.to_sdfg(strict=True)
    sdfg.apply_transformations(GPUTransformSDFG, options = {'strict_transform':False})

    map_ = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n,nodes.MapEntry))

    add_gpu_location(sdfg, map_, 0)

    # clone GPU scalar
    inodename = 'alpha'
    inode = sdfg.arrays['alpha']
    newdesc = inode.clone()
    newdesc.location = {'gpu':1}
    newdesc.storage = StorageType.GPU_Global
    newdesc.transient = True
    name = sdfg.add_datadesc('gpu_' + inodename,
                                newdesc,
                                find_new_name=True)
    # Replace original scalar
    for state in sdfg.nodes():
            for node in state.nodes():
                if (isinstance(node, nodes.AccessNode)
                        and node.data == inodename):
                    node.data = name
    # Replace memlets
    for state in sdfg.nodes():
        for edge in state.edges():
            if edge.data.data == inodename:
                edge.data.data = name
    
    # add GPU scalar to the copyin state
    copyin_state = sdfg.start_state

    src_array = nodes.AccessNode(inodename, debuginfo=inode.debuginfo)
    dst_array = nodes.AccessNode(name,
                                    debuginfo=inode.debuginfo)
    copyin_state.add_node(src_array)
    copyin_state.add_node(dst_array)
    copyin_state.add_nedge(
        src_array, dst_array,
        Memlet.from_array(src_array.data, src_array.desc(sdfg)))

    sdfg.apply_strict_transformations()

    a_times_X = sdfg(X=X, alpha=alpha, N=n)
    res = X*alpha
    idx = zip(*np.where(~np.isclose(res, X*alpha, atol=0, rtol=1e-7)))
    for i in idx:
        print(i, res[i], X[i]*alpha, X[i], alpha)
    assert np.allclose(res, X*alpha)
    print('PASS')

    # program_objects = sdfg.generate_code()

    # from dace.codegen import compiler
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   sdfg.build_folder)
    # sdfg.view()


if __name__ == "__main__":
    gpu_dma_test()