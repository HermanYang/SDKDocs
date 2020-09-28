from sdk2.graph.run_graph import graph_runner
from sdk2.graph.transform_graph.graph_transformers import graph_transform
from sdk2.graph.transform_graph.node_transformers import node_transform_map
from sdk2.proto import graph_types_pb2, lgf_pb2, node_filters, ops_pb2


class MarkSupportedConstNodes(graph_transform.GraphTransform):

    def __init__(self, hw_specs, sw_config, sim_params):
        """
        Params:
            hw_specs: a hw_spec_pb2.HardwareSpecs() protobuf
            sw_config: a sw_config_pb2.SoftwareConfig() protobuf
            sim_params: a sim_params_pb2.SimulationParams() protobuf
        """
        self._hw_specs = hw_specs
        self._sw_config = sw_config
        self._sim_params = sim_params

    @staticmethod
    def _is_unsupported_const(node):
        if node.HasField(lgf_pb2.LNF.original.DESCRIPTOR.name):
            return node.original.op == ops_pb2.CONST
        return False

    def get_transforms(self, light_graph):
        """
        Returns the transforms to mark const nodes as supported
        if all their outputs go to supported nodes.
        """
        # Initialize transforms
        transforms = []

        # Get the const node transformer if we need it
        graph_type = graph_runner.GraphRunner.get_graph_type(light_graph)
        if graph_type == graph_types_pb2.LGFProtobuf:
            return transforms
        const_transform = node_transform_map.get_node_transform(
            graph_type, ops_pb2.CONST)(self._hw_specs, self._sw_config, self._sim_params)

        # If a constant node has all supported outputs, convert it to a
        # supported constant
        supported_filter = node_filters.supported_node_filter()
        for node in light_graph.nodes():
            if self._is_unsupported_const(node):
                # Node is unsupported constant, check outputs
                is_supported = True
                for outname in light_graph.get_output_node_names_of_node(node):
                    if not supported_filter.matches(
                            light_graph.get_node_by_name(outname), light_graph):
                        # Found unsupported output
                        is_supported = False

                # Update node
                if is_supported:
                    transforms.append(const_transform.transform(node, light_graph))

        return transforms
