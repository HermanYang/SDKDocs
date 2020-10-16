from lt_sdk.graph.graph_collections import graph_collection
from lt_sdk.graph.transform_graph.graph_transformers import collapse_supported_subgraphs


class ExportGraph(object):
    """An interface for exporting graphs"""

    def __init__(self, light_graph, hw_specs, sw_config, sim_params, graph_coll=None):
        """
        Params:
            light_graph: a LightGraph object
            hw_specs: a hw_specs_pb2.HardwareSpecs() protobuf
            sw_config: a sw_config_pb2.SoftwareConfig() protobuf
            sim_params: a sim_params_pb2.SimulationParams() protobuf
            graph_coll: a graph_collection.GraphCollection() object paired with the
                given light_graph
        """
        self._light_graph = light_graph
        self._hw_specs = hw_specs
        self._sw_config = sw_config
        self._sim_params = sim_params
        self._graph_coll = graph_coll or graph_collection.NullGraphCollection()

    @staticmethod
    def get_collapsed_light_graph(light_graph):
        """Collapses the supported subraphs in the given light_graph"""
        return collapse_supported_subgraphs.CollapseSupportedSubgraphs().\
            process_transforms(light_graph)

    def export_graph(self, graph_path):
        """
        Params:
            graph_path: path to a file or directory storing an exported graph

        Exports self._light_graph to a new graph version an stores in in graph_path
        """
        raise NotImplementedError()
