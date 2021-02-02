# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains the GPUMultiTransformMap transformation. """

from dace import dtypes, registry
from dace.data import Scalar
from dace.sdfg import has_dynamic_map_inputs
from dace.sdfg import utils as sdutil
from dace.sdfg import nodes
from dace.properties import make_properties, Property, SymbolicProperty
from dace.symbolic import simplify_ext
from dace.transformation import transformation
from dace.properties import make_properties, set_property_from_string

# There is probably a better way to do this
from dace.config import Config


@registry.autoregister_params(singlestate=True)
@make_properties
class GPUMultiTransformMap(transformation.Transformation):

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    new_dim_prefix = Property(dtype=str,
                              default="gpu",
                              allow_none=True,
                              desc="Prefix for new dimension name")

    number_of_gpus = SymbolicProperty(
        default=None,
        allow_none=True,
        desc="number of gpus to divide the map onto,"
        " if not used, uses the amount specified"
        " in the dace.config in max_number_gpus.")

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
        if (Config.get("compiler", "cuda", "max_number_gpus") < 2):
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

        inner_map_entry = graph.nodes()[self.subgraph[
            GPUMultiTransformMap._map_entry]]

        number_of_gpus = self.number_of_gpus
        ngpus = Config.get("compiler", "cuda", "max_number_gpus")
        if (number_of_gpus == None):
            number_of_gpus = ngpus
        if number_of_gpus > ngpus:
            raise ValueError(
                'Requesting more gpus than specified in the dace config')

        # Avoiding import loops
        from dace.transformation.dataflow.strip_mining import StripMining
        from dace.transformation.dataflow.local_storage import LocalStorage

        outer_map = StripMining.apply_to(sdfg,
                                         dict(
                                             dim_idx=-1,
                                             new_dim_prefix=self.new_dim_prefix,
                                             tile_size=number_of_gpus,
                                             tiling_type='number_of_tiles'),
                                         _map_entry=inner_map_entry)

        # Find all in-edges that lead to candidate[GPUMultiTransformMap._map_entry]
        inner_edges = [
            e for e in graph.in_edges(inner_map_entry)
            if isinstance(e.src, nodes.EntryNode)
        ]
        outer_map_entry = inner_edges[0].src

        inner_map_exit = graph.exit_node(inner_map_entry)
        outer_map_exit = graph.exit_node(outer_map_entry)

        gpu_id = outer_map_entry.params[0]
        inner_map_entry.map.location = {"gpu": gpu_id}
        inner_map_entry.map._schedule = dtypes.ScheduleType.GPU_Device

        # Add multi GPU schedule attribute to outer map
        outer_map_entry.map._schedule = dtypes.ScheduleType.GPU_Multiple

        # Add location to the tasklet
        tasklet = None
        edges = [
            e for e in graph.in_edges(inner_map_exit)
            if isinstance(e.src, nodes.Tasklet)
        ]
        tasklet = edges[0].src
        tasklet.location = {"gpu": gpu_id}

        # Now create a transient for each array
        in_data = []
        for e in graph.in_edges(outer_map_entry):
            node = e.src
            if isinstance(node, nodes.AccessNode) and not isinstance(
                    node.desc(sdfg), Scalar):
                in_data.append(node.data)
        prefix = "gpu_multi_"
        out_data = []
        for data_name in in_data:
            in_data_node = LocalStorage.apply_to(sdfg,
                                                 dict(array=data_name,
                                                      prefix=prefix),
                                                 verify=False,
                                                 save=False,
                                                 node_a=outer_map_entry,
                                                 node_b=inner_map_entry)
            in_data = in_data_node.desc(sdfg)
            in_data.location = {"gpu": gpu_id}
            in_data.storage = dtypes.StorageType.GPU_Global

        for e in graph.out_edges(outer_map_exit):
            node = e.dst
            if isinstance(node, nodes.AccessNode) and not isinstance(
                    node.desc(sdfg), Scalar):
                data_name = node.data
                if prefix + data_name in sdfg.arrays:
                    out_data.append((data_name, False))
                else:
                    out_data.append((data_name, True))

        for data_name, create_array in out_data:
            out_data_node = LocalStorage.apply_to(
                sdfg,
                dict(array=data_name, prefix=prefix, create_array=create_array),
                verify=False,
                save=False,
                node_a=inner_map_exit,
                node_b=outer_map_exit)
            out_data_node.desc(sdfg).location = {"gpu": gpu_id}
            out_data_node.desc(sdfg).storage = dtypes.StorageType.GPU_Global
