from sdk2.proto import node_filters


class ImportGraph(object):
    """An interface for importing graphs"""

    def __init__(self, graph_path, sw_config, input_edges=None):
        """
        Params:
            graph_path: path to a file or directory storing the original graph
            sw_config: a sw_config_pb2.SoftwareConfig() protobuf
            input_edges: A list of EdgeInfo objects that specify the input edges
                extracted from some calibration/input data that will be used to run
                the graph. This helps with determining shapes for graphs that might
                support unknown dimensions.
        """
        self._graph_path = graph_path
        self._sw_config = sw_config
        self._ignore_nodes_filter = node_filters.NodeFilter(
            self._sw_config.ignore_nodes_filter)
        self._input_edges = None
        if input_edges:
            self._input_edges = {(x.name, x.port): x for x in input_edges}

            # This could happen if name is unset for multiple inputs.
            if len(input_edges) != len(self._input_edges):
                raise ValueError(
                    "Duplicate name-port given in input edges: {0}".format(input_edges))

    def as_light_graph(self):
        """Returns the initial LightGraph corresponding to the original imported graph"""
        raise NotImplementedError()
