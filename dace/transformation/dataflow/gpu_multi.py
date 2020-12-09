# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains the GPUMultiTransformMap transformation. """

from dace import dtypes, registry
from dace.sdfg import has_dynamic_map_inputs
from dace.sdfg import utils as sdutil
from dace.sdfg import nodes
from dace.transformation import transformation
from dace.properties import make_properties, set_property_from_string

# There is probably a better way to do this
from dace.config import Config


@registry.autoregister_params(singlestate=True)
@make_properties
class GPUMultiTransformMap(transformation.Transformation):

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(GPUMultiTransformMap._map_entry)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        map_entry = graph.nodes()[candidate[GPUMultiTransformMap._map_entry]]

        # Check if there is more than one GPU available:
        if(Config.get("compiler", "cuda", "max_number_gpus") < 2):
            return False

        # Check if the map is one-dimensional
        if map_entry.map.range.dims() != 1:
            return False

        # We cannot transform a map which is already of schedule type GPU_Multi
        if map_entry.map.schedule == dtypes.ScheduleType.GPU_Multiple:
            return False

        # We cannot transform a map which is already inside a GPU map, or in
        # another device
        schedule_whitelist = [
            dtypes.ScheduleType.Default, dtypes.ScheduleType.Sequential
        ]
        sdict = graph.scope_dict()
        parent = sdict[map_entry]
        while parent is not None:
            if parent.map.schedule not in schedule_whitelist:
                return False
            parent = sdict[parent]

        # Dynamic map ranges not supported (will allocate dynamic memory)
        if has_dynamic_map_inputs(graph, map_entry):
            return False

        # MPI schedules currently do not support WCR
        map_exit = graph.exit_node(map_entry)
        if any(e.data.wcr for e in graph.out_edges(map_exit)):
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = graph.nodes()[candidate[GPUMultiTransformMap._map_entry]]

        return map_entry.map.label

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]

        map_entry = graph.nodes()[self.subgraph[GPUMultiTransformMap._map_entry]]
        num_gpus = Config.get("compiler", "cuda", "max_number_gpus")

        # Avoiding import loops
        from dace.transformation.dataflow.strip_mining import StripMining
        from dace.transformation.dataflow.local_storage import LocalStorage

        rangeexpr = map_entry.map.range.num_elements()

        maptiling_subgraph = {
            StripMining._map_entry: self.subgraph[GPUMultiTransformMap._map_entry]
        }
        sdfg_id = sdfg.sdfg_id
        stripmine = StripMining(sdfg_id, self.state_id, maptiling_subgraph,
                              self.expr_index)
        stripmine.new_dim_prefix = "gpu"
        stripmine.tile_size = str(num_gpus)
        stripmine.inner_map = False
        stripmine.apply(sdfg)


        # Find all in-edges that lead to candidate[GPUMultiTransformMap._map_entry]
        outer_map = None
        edges = [
            e for e in graph.in_edges(map_entry)
            if isinstance(e.src, nodes.EntryNode)
        ]

        outer_map = edges[0].src
        
        gpu_id = outer_map.params[0]
        map_entry.map.location={"gpu":gpu_id}
        map_entry.map._schedule = dtypes.ScheduleType.GPU_Device

        # Add MPI schedule attribute to outer map
        outer_map.map._schedule = dtypes.ScheduleType.GPU_Multiple
        # Now create a transient for each array
        for e in edges:
            # LocalStorage.apply_to()
            in_local_storage_subgraph = {
                LocalStorage.node_a: graph.node_id(outer_map),
                LocalStorage.node_b: self.subgraph[GPUMultiTransformMap._map_entry]
            }

            sdfg_id = sdfg.sdfg_id
            in_local_storage = LocalStorage(sdfg_id, self.state_id,
                                            in_local_storage_subgraph,
                                            self.expr_index)
            in_local_storage.array = e.data.data
            in_data_node=in_local_storage.apply(sdfg)
            in_data_node.desc(sdfg).location={"gpu":gpu_id}
            in_data_node.desc(sdfg).storage=dtypes.StorageType.GPU_Global
            

        # Transform OutLocalStorage for each output of the MPI map
        in_map_exit = graph.exit_node(map_entry)
        out_map_exit = graph.exit_node(outer_map)

        tasklet = None
        edges = [
            e for e in graph.in_edges(in_map_exit)
            if isinstance(e.src, nodes.Tasklet)
        ]
        tasklet = edges[0].src
        tasklet.location={"gpu":gpu_id}

        for e in graph.out_edges(out_map_exit):
            name = e.data.data
            outlocalstorage_subgraph = {
                LocalStorage.node_a: graph.node_id(in_map_exit),
                LocalStorage.node_b: graph.node_id(out_map_exit)
            }
            sdfg_id = sdfg.sdfg_id
            outlocalstorage = LocalStorage(sdfg_id, self.state_id,
                                           outlocalstorage_subgraph,
                                           self.expr_index)
            outlocalstorage.array = name
            out_data_node = outlocalstorage.apply(sdfg)
            out_data_node.desc(sdfg).location={"gpu":gpu_id}
            out_data_node.desc(sdfg).storage=dtypes.StorageType.GPU_Global
