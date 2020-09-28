from sdk2.graph.transform_graph.graph_transformers import (apply_node_map,
                                                           collapse_supported_subgraphs)
from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    base_transform
from sdk2.proto import lgf_pb2, node_filters, transform_result_pb2


class ExpandSubgraphNodesNodeTransform(base_transform.BaseTransform):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._node_name_to_subgraph_id = {}

    def node_name_to_subgraph_id(self, node_name):
        return self._node_name_to_subgraph_id.get(node_name, None)

    def transform(self, subgraph_node, light_graph):
        to_add = []
        to_reroute = []
        to_output_swap = []
        subgraph_id = (
            collapse_supported_subgraphs.CollapseSupportedSubgraphs.get_subgraph_index(
                subgraph_node.name))

        # Map the subgraph graph inputs to the subgraph node inputs
        input_edge_map = {}
        for i, graph_inp in enumerate(subgraph_node.subgraph.graph.input_edges):
            input_edge_map[(graph_inp.name, graph_inp.port)] = subgraph_node.inputs[i]

        for node in subgraph_node.subgraph.graph.nodes:
            self._node_name_to_subgraph_id[node.name] = subgraph_id

            # Update node inputs if necessary
            for e_in in node.inputs:
                key = (e_in.name, e_in.port)
                if key in input_edge_map:
                    e_in.CopyFrom(input_edge_map[key])

            # Add all control inputs to every node in the subgraph
            node.control_inputs.extend(subgraph_node.control_inputs)

            to_add.append(node)

        for old_edge in subgraph_node.outputs:
            # Reroute output edges of the subgraph
            new_edge = subgraph_node.subgraph.graph.output_edges[old_edge.port]
            to_reroute.append(
                (transform_result_pb2.ToReroute.edge_reroute.DESCRIPTOR.name, [],
                 old_edge, new_edge))

            # Control inputs depending on the subgraph node now depend on
            # all outputs of the subgraph
            to_reroute.append(
                (transform_result_pb2.ToReroute.control_input_reroute.DESCRIPTOR.name,
                 [], [subgraph_node.name
                     ], [e.name for e in subgraph_node.subgraph.graph.output_edges] +
                 [nn for nn in subgraph_node.subgraph.graph.output_node_names]))

        if subgraph_node.name in light_graph.output_node_names():
            # If the subgraph node was an output node of the light graph,
            # make its output nodes the output nodes of the expanded light graph
            to_output_swap.append(([subgraph_node.name],
                                   list(subgraph_node.subgraph.graph.output_node_names)))

        return self.create_transform_result(to_add=to_add,
                                            to_reroute=to_reroute,
                                            to_output_swap=to_output_swap)


class ExpandSubgraphNodes(apply_node_map.ApplyNodeMap):

    def __init__(self, hw_specs, sw_config, sim_params):
        self._node_transform = ExpandSubgraphNodesNodeTransform(
            hw_specs, sw_config, sim_params)
        subgraph_node_filter = node_filters.which_oneof_filter(
            lgf_pb2.LNF.subgraph.DESCRIPTOR.name)
        super().__init__(hw_specs, sw_config,
                         {subgraph_node_filter: self._node_transform})

    def node_name_to_subgraph_id(self, node_name):
        return self._node_transform.node_name_to_subgraph_id(node_name)
