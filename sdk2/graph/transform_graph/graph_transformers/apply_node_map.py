import logging

from sdk2.graph.transform_graph.graph_transformers import graph_transform
from sdk2.graph.transform_graph.node_transformers import node_transform_map
from sdk2.proto import node_filters


class ApplyNodeMap(graph_transform.GraphTransform):

    def __init__(self, hw_specs, sw_config, node_map):
        """
        Params:
            hw_specs: a hw_spec_pb2.HardwareSpecs() protobuf
            sw_config: a sw_config_pb2.SoftwareConfig() protobuf
            node_map: A map of {NodeFilter -> NodeTransform}
        """
        self._hw_specs = hw_specs
        self._sw_config = sw_config
        self._node_map = node_map

    def get_transforms(self, light_graph):
        """
        Returns the transforms found from the self.node_map. Specifically, for each
        filter in self.node_map, for any node where the filter matches that node,
        add the transforms from self.node_map[filter].transform(node, light_graph)
        """
        filter_nodes = {filt: [] for filt in self._node_map.keys()}
        for node in light_graph.nodes():
            matched = False
            for filt in self._node_map.keys():
                if filt.matches(node, light_graph):
                    if matched:
                        raise ValueError("node {0} matches more than one filter.".format(
                            node.name))
                    filter_nodes[filt].append(node)
                    matched = True

        results = []
        for filt, nodes in filter_nodes.items():
            for node in nodes:
                if self._node_map[filt].can_transform(node, light_graph):
                    results.append(self._node_map[filt].transform(node, light_graph))
                else:
                    logging.warning("Not transforming {0}".format(node.name))

        return results

    @staticmethod
    def get_node_map_from_filter_transform_map(filter_transform_map,
                                               *node_transform_args,
                                               **node_transform_kwargs):
        """
        Params:
            node_transform_config: a list of sw_config_pb2.FilterTransformPair() objects
            node_transform_args: arguments for NodeTransform objects
            node_transform_kwargs: keyword arguments for NodeTransform objects

        Returns:
            node_map: a dictionary that maps node filters to node transforms
        """
        return {
            node_filters.NodeFilter(pair.filter): node_transform_map.get_node_transform(
                pair.transform.graph_type, pair.transform.op)(*node_transform_args,
                                                              *node_transform_kwargs)
            for pair in filter_transform_map
        }

    @staticmethod
    def get_opu_node_filter(sw_config):
        return node_filters.which_oneof_filter(*(sw_config.node_types.opu_nodes))
