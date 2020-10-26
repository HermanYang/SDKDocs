from lt_sdk.graph import lgf_graph
from lt_sdk.graph.transform_graph import utils
from lt_sdk.proto import lgf_pb2, transform_result_pb2


class GraphTransform(object):
    """Interface for a graph transform object"""

    def __init__(self):
        pass

    def get_transforms(self, light_graph):
        """Returns a list of transform_result_pb2.TransformResult objects"""
        raise NotImplementedError()

    def get_meta_graph_info(self, meta_graph_info):
        """Override this function to update the meta graph info"""
        return meta_graph_info

    @staticmethod
    def _edge_in_list(edge, edge_list):
        for e in edge_list:
            if utils.edges_match(edge, e):
                return True

        return False

    @staticmethod
    def _update_input_edges(nodes, input_edges):
        node_names = {n.name for n in nodes}
        for node in nodes:
            for e_in in node.inputs:
                if (e_in.name not in node_names
                        and not (GraphTransform._edge_in_list(e_in,
                                                              input_edges))):
                    new_inp = lgf_pb2.EdgeInfo()
                    new_inp.CopyFrom(e_in)
                    input_edges.append(new_inp)

    @staticmethod
    def _add_nodes(to_add, nodes, input_edges, output_edges):
        node_names = {n.name for n in nodes}
        for transform in to_add:
            new_node = transform.node
            if new_node.name not in node_names:
                nodes.append(new_node)
                node_names.add(new_node.name)

        GraphTransform._update_input_edges(nodes, input_edges)

    @staticmethod
    def _update_edges_replace(new_node, edges_to_check):
        for e_new in new_node.outputs:
            for e_old in edges_to_check:
                if utils.edges_match(e_new, e_old):
                    e_old.CopyFrom(e_new)

    @staticmethod
    def _replace_nodes(to_replace, nodes, input_edges, output_edges):
        node_dict = {n.name: n for n in nodes}
        for transform in to_replace:
            new_node = transform.node
            if new_node.name in node_dict:
                node_dict[new_node.name].CopyFrom(new_node)

            GraphTransform._update_edges_replace(new_node, input_edges)
            GraphTransform._update_edges_replace(new_node, output_edges)

    @staticmethod
    def _update_edges_reroute(old_edge, new_edge, edges_to_check):
        for e in edges_to_check:
            if utils.edges_match(old_edge, e):
                e.CopyFrom(new_edge)

    @staticmethod
    def _edge_reroute(nodes_to_check, old_edge, new_edge, output_edges):
        for node in nodes_to_check:
            if new_edge.name == node.name:
                # new_edge is an output of node
                continue

            GraphTransform._update_edges_reroute(old_edge, new_edge, node.inputs)

        # Reroute can only change output_edges of the graph
        GraphTransform._update_edges_reroute(old_edge, new_edge, output_edges)

    @staticmethod
    def _control_input_reroute(nodes_to_check, old_node_names, new_node_names):
        old_node_names = set(old_node_names)
        new_node_names = set(new_node_names)

        for node in nodes_to_check:
            control_inputs = set(node.control_inputs)
            if old_node_names.issubset(control_inputs):
                control_inputs.difference_update(old_node_names)
                control_inputs.update(new_node_names)
                node.control_inputs[:] = sorted(control_inputs)

    @staticmethod
    def _reroute_nodes(to_reroute, nodes, input_edges, output_edges):
        node_dict = {n.name: n for n in nodes}
        for transform in to_reroute:
            # Get nodes to check
            if len(transform.dst_node_names) == 0:
                nodes_to_check = nodes
            else:
                nodes_to_check = [
                    node_dict[node_name] for node_name in transform.dst_node_names
                ]

            # Different types of reroute procedures
            if transform.HasField(
                    transform_result_pb2.ToReroute.edge_reroute.DESCRIPTOR.name):
                GraphTransform._edge_reroute(nodes_to_check,
                                             transform.edge_reroute.old_edge,
                                             transform.edge_reroute.new_edge,
                                             output_edges)
            elif transform.HasField(transform_result_pb2.ToReroute.control_input_reroute.
                                    DESCRIPTOR.name):
                GraphTransform._control_input_reroute(
                    nodes_to_check,
                    transform.control_input_reroute.old_node_names,
                    transform.control_input_reroute.new_node_names)
            else:
                raise ValueError("Invalid ToReroute transform: {}".format(transform))

    @staticmethod
    def _output_swap(to_output_swap, output_node_names):
        for transform in to_output_swap:
            for node_name in transform.old_output_node_names:
                output_node_names.remove(node_name)

            output_node_names.extend(transform.new_output_node_names)

    @staticmethod
    def concat_transforms(transforms):
        """
        Converts a list of transform_result_pb2.TransformResult() protobufs
        into a single transform_result_pb2.TransformResult
        """
        result = transform_result_pb2.TransformResult()
        for transform in transforms:
            result.to_add.extend(transform.to_add)
            result.to_replace.extend(transform.to_replace)
            result.to_reroute.extend(transform.to_reroute)
            result.to_output_swap.extend(transform.to_output_swap)

        return result

    def process_transforms(self, light_graph, prune=True):
        """
        Returns a new LightGraph object which is the result of performing
        the transormations returned from self.get_transforms(light_graph)
        on the given light_graph
        """
        transforms = self.concat_transforms(self.get_transforms(light_graph))

        # Remember light_graph is immutable, so we will mutate nodes,
        # input_edges, output_edges, then create a new light_graph
        nodes = light_graph.nodes()
        input_edges = light_graph.input_edges()
        output_edges = light_graph.output_edges()
        output_node_names = light_graph.output_node_names()
        meta_graph_info = self.get_meta_graph_info(light_graph.meta_graph_info())

        self._add_nodes(transforms.to_add, nodes, input_edges, output_edges)
        self._replace_nodes(transforms.to_replace, nodes, input_edges, output_edges)
        self._reroute_nodes(transforms.to_reroute, nodes, input_edges, output_edges)
        self._output_swap(transforms.to_output_swap, output_node_names)

        # Create transformed graph and prune it
        transformed_graph = lgf_graph.LightGraph(nodes,
                                                 input_edges=input_edges,
                                                 output_edges=output_edges,
                                                 output_node_names=output_node_names,
                                                 meta_graph_info=meta_graph_info)
        if prune:
            transformed_graph = transformed_graph.prune_graph()

        return transformed_graph
