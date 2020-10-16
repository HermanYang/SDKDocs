import logging

from lt_sdk.graph import lgf_graph
from lt_sdk.graph.transform_graph.graph_transformers import graph_transform
from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    base_transform,
)
from lt_sdk.proto import lgf_pb2, node_filters, transform_result_pb2


class CyclicGraphError(Exception):
    """Exception raised when collasping supported subgraphs results in a cycle
    in the transformed light graph."""


class CollapseSupportedSubgraphs(graph_transform.GraphTransform):

    _NODE_NAME_PREFIX = "LGFSubgraph_"

    @staticmethod
    def get_subgraph_node_name(subgraph_index):
        return "{}{}".format(CollapseSupportedSubgraphs._NODE_NAME_PREFIX,
                             subgraph_index)

    @staticmethod
    def get_subgraph_index(subgraph_node_name):
        assert (subgraph_node_name.startswith(
            CollapseSupportedSubgraphs._NODE_NAME_PREFIX))
        return int(
            subgraph_node_name[len(CollapseSupportedSubgraphs._NODE_NAME_PREFIX):])

    @staticmethod
    def _get_next_subgraph_index(light_graph):
        max_subgraph_index = -1
        for node in light_graph.nodes():
            if node.name.startswith(CollapseSupportedSubgraphs._NODE_NAME_PREFIX):
                max_subgraph_index = max(
                    max_subgraph_index,
                    CollapseSupportedSubgraphs.get_subgraph_index(node.name))

        return max_subgraph_index + 1

    def _subgraph_nodes_to_subgraph(self, subgraph_nodes, light_graph):
        """
        Params:
            subgraph_nodes: a list of nodes that form a subgraph of supported nodes
            light_graph: original light graph the subgraph_nodes were extracted from

        Returns:
            subgraph: a LightGraph object for the nodes in subgraph_nodes
        """
        subgraph_node_names = {n.name for n in subgraph_nodes}
        original_output_node_names = set(light_graph.output_node_names())
        light_graph_output_edges = light_graph.output_edges()
        # Create a list of edges that nodes outside the subgraph needs.
        # Might contain duplicates
        inputs_for_other_nodes = []
        for node in light_graph.nodes():
            if node.name not in subgraph_node_names:
                inputs_for_other_nodes.extend(node.inputs)

        # Get the inputs and outputs of the subgraph
        input_edges = []
        output_edges = []
        output_node_names = []
        control_inputs = set()

        for node in subgraph_nodes:
            # If a node's inputs need an edge that is not found in the subgraph, it
            # must be an input to the subgraph
            for e in node.inputs:
                if (e.name not in subgraph_node_names
                        and not (self._edge_in_list(e,
                                                    input_edges))):
                    input_edges.append(e)

            # If a node has a control input that is not found in the subgraph, it
            # must be a control input to the subgraph
            for inp_name in node.control_inputs:
                if inp_name not in subgraph_node_names:
                    control_inputs.add(inp_name)

            # If a node's outputs have an edge that is an output of light_graph or
            # an edge that a node outside the subgraph needs, then it must be an output
            # of the subgraph
            for e in node.outputs:
                if (self._edge_in_list(e,
                                       light_graph_output_edges)
                        or self._edge_in_list(e,
                                              inputs_for_other_nodes)):
                    if not self._edge_in_list(e, output_edges):
                        output_edges.append(e)

            # If a node was an output node in the original graph, then it must
            # also be an output node of the subgraph
            if node.name in original_output_node_names:
                output_node_names.append(node.name)

        subgraph = lgf_graph.LightGraph(subgraph_nodes,
                                        input_edges=input_edges,
                                        output_edges=output_edges,
                                        output_node_names=output_node_names)
        return subgraph, control_inputs

    def _get_unsupported_path_dict_for_node(self,
                                            node,
                                            node_cache,
                                            unsupported_path_dicts):
        """
        Recursively constructs and returns a dict for a given node that maps any src_node
        to a boolean value which indicates whether there is at least one unsupported node
        along the path from src_node to node.
        """
        assert node.name in node_cache

        if node.name not in unsupported_path_dicts:
            unsupported_path_dicts[node.name] = {}

            is_input_node = True
            for e in node.inputs:
                if e.name in node_cache:
                    is_input_node = False

            if not is_input_node:

                all_paths = []

                for e in node.inputs:
                    if e.name not in node_cache:
                        continue
                    new_path = {}
                    new_path[e.name] = not (node_cache[node.name].supported
                                            and node_cache[e.name].supported)
                    node_path_dict = self._get_unsupported_path_dict_for_node(
                        node_cache[e.name],
                        node_cache,
                        unsupported_path_dicts)
                    for name, status in node_path_dict.items():
                        new_path[name] = new_path[e.name] or status

                    all_paths.append(new_path)

                for path in all_paths:
                    for k, val in path.items():
                        unsupported_path_dicts[
                            node.name][k] = unsupported_path_dicts[node.name].get(
                                k,
                                False) or val

        return unsupported_path_dicts[node.name]

    def _unsupported_path_exists(self,
                                 src_node,
                                 dst_node,
                                 node_cache,
                                 unsupported_path_dicts):
        """
        Returns True if there is a path src_node --> dst_node that has at
        least 1 unsupported node.

        See docstring of _safe_to_add_node_to_subgraph for more details
        about unsupported_path_dicts.
        """
        if src_node.name == dst_node.name:
            return not node_cache[dst_node.name].supported

        return self._get_unsupported_path_dict_for_node(
            dst_node,
            node_cache,
            unsupported_path_dicts,
        ).get(src_node.name,
              False)

    def _safe_to_add_node_to_subgraph(self,
                                      candidate_node,
                                      new_subgraph_nodes,
                                      node_cache,
                                      unsupported_path_dicts):
        """
        Params:
            candidate_node: a supported node that we are trying to add to the
                new subgraph
            new_subgraph_nodes: a list of nodes in the new subgraph
            node_cache: A dict of {node name -> node} to avoid lots of copying
                from light_graph.get_node_by_name
            unsupported_path_dicts: a dict mapping a dst_node name to a dict, which
                then maps a src_node name to a boolean value indicating whether
                there is at least one unsupported node along the paths from
                src_node to dst_node. In addition, if a given node n is not in
                unsupported_path_dicts[dst_node.name], that means dst_node is not
                accessible from n, thus there is no unsupported node between
                n and dst_node.

        Returns:
            True if it is safe to add candidate_node to new_subgraph_nodes,
                False otherwise
        """

        # If there is a path in light_graph between the candidate_node and any node in
        # new_subgraph_nodes that goes through an unsupported node, then it is NOT
        # safe to add candidate_node to new_subgraph_nodes
        for node in new_subgraph_nodes:
            if (self._unsupported_path_exists(candidate_node,
                                              node,
                                              node_cache,
                                              unsupported_path_dicts)
                    or self._unsupported_path_exists(node,
                                                     candidate_node,
                                                     node_cache,
                                                     unsupported_path_dicts)):
                return False

        return True

    def _get_supported_subgraph_lists(self, light_graph):
        """
        Collapse supported nodes in a light_graph to a list of subgraphs.
        """
        light_graph_nodes = light_graph.nodes()
        node_cache = {n.name: n for n in light_graph_nodes}
        # Extract subgraphs from light_graph
        supported_filter = node_filters.supported_node_filter()
        visited_node_names = set()
        subgraphs = []
        unsupported_path_dicts = {}

        for node in light_graph_nodes:
            if (supported_filter.matches(node,
                                         light_graph)
                    and node.name not in visited_node_names):
                # Found a node that is in an undiscovered subgraph
                new_subgraph_nodes = []
                for candidate_node in light_graph.bfs(node,
                                                      bidirectional=True,
                                                      node_filter=supported_filter):
                    if (candidate_node.name not in visited_node_names
                            and self._safe_to_add_node_to_subgraph(
                                candidate_node,
                                new_subgraph_nodes,
                                node_cache,
                                unsupported_path_dicts)):
                        new_subgraph_nodes.append(candidate_node)
                        visited_node_names.add(candidate_node.name)
                subgraphs.append(
                    self._subgraph_nodes_to_subgraph(new_subgraph_nodes,
                                                     light_graph))

        if self._is_cyclic(subgraphs, light_graph):
            raise CyclicGraphError(
                "Cycles detected after collapsing supported subgraphs")

        return subgraphs

    def _is_cyclic(self, subgraphs, light_graph):
        """
        Check if collapsing subgraphs results in cycles in the exported TF graph.
        """
        node_cache = {n.name: n for n in light_graph.nodes() if not n.supported}
        subgraph_name_format = "subgraph_{}"
        for i, (sg, _) in enumerate(subgraphs):
            node_cache[subgraph_name_format.format(i)] = sg

        output_edge_map = {(e.name,
                            e.port): subgraph_name_format.format(i) for i,
                           (sg,
                            _) in enumerate(subgraphs) for e in sg.output_edges()}
        light_graph_input_edges = light_graph.input_edges()

        def check_cyclic_utils(node_name):
            visited_node_names.add(node_name)
            dfs_node_stack.append(node_name)
            node_on_stack[node_name] = True

            node = node_cache[node_name]
            input_edges = node.input_edges() if isinstance(
                node,
                lgf_graph.LightGraph) else node.inputs
            for e in input_edges:
                if (e.name, e.port) in output_edge_map:
                    new_node_name = output_edge_map[(e.name, e.port)]
                elif e.name in node_cache:
                    new_node_name = e.name
                else:
                    assert self._edge_in_list(e, light_graph_input_edges)
                    continue

                if new_node_name not in visited_node_names:
                    if check_cyclic_utils(new_node_name):
                        return True
                elif node_on_stack[new_node_name]:
                    indx = dfs_node_stack.index(new_node_name)

                    # It's possible that we have found a while loop in the graph
                    all_while_nodes = True
                    for node in dfs_node_stack[indx:]:
                        # We only want loops that encompass our subgraph nodes
                        if (isinstance(node,
                                       lgf_graph.LightGraph)) or "while" not in node:
                            all_while_nodes = False

                    if not all_while_nodes:
                        logging.error("Found cycles in transformed graphs: %s",
                                      dfs_node_stack[indx:] + [new_node_name])
                        for n in dfs_node_stack[indx:]:
                            subgraph = node_cache[n]
                            if isinstance(subgraph, lgf_graph.LightGraph):
                                logging.error(
                                    "%s\n%s:\n\tnodes %s\n\tinput edges %s\n"
                                    "\toutput edges %s\n%s",
                                    "*" * 20,
                                    n,
                                    [v.name for v in subgraph.nodes()],
                                    [v.name for v in subgraph.input_edges()],
                                    [v.name for v in subgraph.output_edges()],
                                    "*" * 20)
                        return True

            dfs_node_stack.pop()
            node_on_stack[node_name] = False

            return False

        visited_node_names = set()
        # Cycles, if there are any, must be introduced by collapsing supported subgraphs,
        # so we only do dfs on subgraph nodes.
        for i in range(len(subgraphs)):
            dfs_node_stack = []  # For debug purpose
            node_on_stack = {v: False for v in node_cache}
            subgraph_node_name = subgraph_name_format.format(i)
            if subgraph_node_name not in visited_node_names and check_cyclic_utils(
                    subgraph_node_name):
                return True
        return False

    def get_transforms(self, light_graph):
        """
        Returns the transforms to collapse supported subgraphs in
        light_graph into single nodes.
        """
        subgraphs = self._get_supported_subgraph_lists(light_graph)

        # Node transformations converting each subgraph into a single node
        to_add = []
        to_reroute = []
        to_output_swap = []
        subgraph_index = self._get_next_subgraph_index(light_graph)
        for subgraph, control_inputs in subgraphs:
            subgraph_node = lgf_pb2.LNF()
            subgraph_node.name = self.get_subgraph_node_name(subgraph_index)
            subgraph_node.supported = False
            subgraph_node.subgraph.SetInParent()
            subgraph_node.subgraph.graph.CopyFrom(subgraph.as_lgf_pb())
            subgraph_node.inputs.extend(subgraph.input_edges())
            subgraph_node.control_inputs.extend(control_inputs)

            for j, old_edge in enumerate(subgraph.output_edges()):
                new_edge = lgf_pb2.EdgeInfo()
                new_edge.CopyFrom(old_edge)
                new_edge.name = subgraph_node.name
                new_edge.port = j
                subgraph_node.outputs.add().CopyFrom(new_edge)

                to_reroute.append(
                    (transform_result_pb2.ToReroute.edge_reroute.DESCRIPTOR.name,
                     [],
                     old_edge,
                     new_edge))

            for old_node in subgraph.nodes():
                to_reroute.append((
                    transform_result_pb2.ToReroute.control_input_reroute.DESCRIPTOR.name,
                    [],
                    [old_node.name],
                    [subgraph_node.name]))

            if len(subgraph.output_node_names()):
                to_output_swap.append((subgraph.output_node_names(),
                                       [subgraph_node.name]))

            to_add.append(subgraph_node)

            subgraph_index += 1

        return [
            base_transform.BaseTransform.create_transform_result(
                to_add=to_add,
                to_reroute=to_reroute,
                to_output_swap=to_output_swap)
        ]
