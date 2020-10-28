from lt_sdk.graph.transform_graph.graph_transformers import graph_transform
from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    base_transform,
)
from lt_sdk.proto import common_pb2, lgf_pb2, transform_result_pb2


class ConvertToActivationScaleCalibrationGraph(graph_transform.GraphTransform):
    """
    Graph transformer that creates an activation scale calibration
    graph by inserting collect hist nodes into a graph
    """

    def __init__(self, nodes_to_calibrate, sw_config, hist_coll):
        self._nodes_to_calibrate = nodes_to_calibrate
        self._sw_config = sw_config
        self._hist_coll = hist_coll
        self._counter = -1
        self._edge_to_key = {}
        self._node_name_to_key = {}

    @staticmethod
    def insert_collect_hist_node(edge, sw_config, hist_key, num_bins, hist_coll):
        to_add = []
        to_reroute = []
        edge_reroute = transform_result_pb2.ToReroute.edge_reroute.DESCRIPTOR.name

        # Cast to float if necessary
        insert_cast = (edge.dtype.p > sw_config.float_type.p)
        if insert_cast:
            # Cast to float
            cast_node = lgf_pb2.LNF()
            cast_node.name = "{}_{}_cast".format(edge.name, edge.port)
            cast_node.supported = True
            cast_node.cast.SetInParent()

            # Cast inputs and outputs
            cast_node.inputs.add().CopyFrom(edge)
            cast_output_edge = lgf_pb2.EdgeInfo()
            cast_output_edge.name = cast_node.name
            cast_output_edge.port = 0
            cast_output_edge.dtype.CopyFrom(sw_config.float_type)
            cast_output_edge.shape.CopyFrom(edge.shape)
            cast_node.outputs.add().CopyFrom(cast_output_edge)

            to_add.append(cast_node)
            to_reroute.append((edge_reroute, [], edge, cast_output_edge))
        else:
            cast_output_edge = edge

        # Collect hist node
        collect_hist_node = lgf_pb2.LNF()
        collect_hist_node.name = "{}_{}_collect_hist".format(edge.name, edge.port)
        collect_hist_node.supported = True
        collect_hist_node.collect_hist.SetInParent()
        collect_hist_node.collect_hist.hist_keys.keys.append(hist_key)
        collect_hist_node.collect_hist.hist_keys.quant_type = common_pb2.QT_SINGLE
        hist_coll.initialize_empty_histogram(hist_key, num_bins)

        # Calibration inputs and outputs
        collect_hist_node.inputs.add().CopyFrom(cast_output_edge)
        collect_hist_output_edge = lgf_pb2.EdgeInfo()
        collect_hist_output_edge.CopyFrom(cast_output_edge)
        collect_hist_output_edge.name = collect_hist_node.name
        collect_hist_output_edge.port = 0
        collect_hist_node.outputs.add().CopyFrom(collect_hist_output_edge)

        to_add.append(collect_hist_node)
        to_reroute.append((edge_reroute, [], cast_output_edge, collect_hist_output_edge))

        # Reverse cast if necessary
        if insert_cast:
            # Reverse cast
            reverse_cast_node = lgf_pb2.LNF()
            reverse_cast_node.name = "{}_{}_reverse_cast".format(edge.name, edge.port)
            reverse_cast_node.supported = True
            reverse_cast_node.cast.SetInParent()

            # Reverse cast inputs and outputs
            reverse_cast_node.inputs.add().CopyFrom(collect_hist_output_edge)
            reverse_cast_output_edge = lgf_pb2.EdgeInfo()
            reverse_cast_output_edge.CopyFrom(collect_hist_output_edge)
            reverse_cast_output_edge.name = reverse_cast_node.name
            reverse_cast_output_edge.dtype.CopyFrom(edge.dtype)
            reverse_cast_node.outputs.add().CopyFrom(reverse_cast_output_edge)

            to_add.append(reverse_cast_node)
            to_reroute.append((edge_reroute,
                               [],
                               collect_hist_output_edge,
                               reverse_cast_output_edge))

        return base_transform.BaseTransform.create_transform_result(
            to_add=to_add,
            to_reroute=to_reroute)

    def _get_edges_to_calibrate(self, light_graph):
        # Assumes data input is always at index 0
        return [
            light_graph.get_node_by_name(node_info.node_name).inputs[0]
            for node_info in self._nodes_to_calibrate
        ]

    def _get_new_hist_key(self, edge_tuple, node_info):
        # Updates for new key
        assert (edge_tuple not in self._edge_to_key)
        self._counter += 1
        self._edge_to_key[edge_tuple] = self._counter
        self._node_name_to_key[node_info.node_name] = self._edge_to_key[edge_tuple]
        return self.get_key_from_node_name(node_info.node_name)

    def get_key_from_node_name(self, node_name):
        return self._node_name_to_key[node_name]

    def _insert_collect_hist_node(self, edge, node_info):
        # There is already a histogram for this edge. This happens if two nodes
        # are quantizing the same input edge
        edge_tuple = (edge.name, edge.port)
        if edge_tuple in self._edge_to_key:
            self._node_name_to_key[node_info.node_name] = self._edge_to_key[edge_tuple]
            return transform_result_pb2.TransformResult()

        # Need to insert a collect hist node for this edge
        return self.insert_collect_hist_node(
            edge,
            self._sw_config,
            self._get_new_hist_key(edge_tuple,
                                   node_info),
            self._sw_config.activation_scale_num_bins,
            self._hist_coll)

    def get_transforms(self, light_graph):
        edges_to_calibrate = self._get_edges_to_calibrate(light_graph)
        return [
            self._insert_collect_hist_node(edges_to_calibrate[i],
                                           self._nodes_to_calibrate[i])
            for i in range(len(edges_to_calibrate))
        ]
