from lt_sdk.proto import lgf_pb2, node_filters, ops_pb2


class LightGraph(object):
    """
    Wrapper around lgf_pb2.LGF() protobuf with some helper functions
    Immutable data type
    """

    CONTROL_FLOW_OPS = {
        ops_pb2.ENTER,
        ops_pb2.SWITCH,
        ops_pb2.MERGE,
        ops_pb2.NEXT_ITERATION,
        ops_pb2.EXIT,
    }

    CONST_OPS = {
        ops_pb2.CONST,
        ops_pb2.VARIABLE,
    }

    CONST_NODES = {
        lgf_pb2.LNF.const.DESCRIPTOR.name,
        lgf_pb2.LNF.variable.DESCRIPTOR.name,
    }

    IS_CONST_ATTR = "is_constant"

    def __init__(self,
                 nodes,
                 input_edges=None,
                 output_edges=None,
                 output_node_names=None,
                 meta_graph_info=None):
        """
        Params:
            subgraphs: a list of lgf_pb2.LNF() protobufs
            input_edges: an optional list of lgf_pb2.EdgeInfo() protobufs
            output_edges: an optional list of lgf_pb2.EdgeInfo() protobufs
            output_nodes: a optional list of strings corresponding to output node names
            meta_graph_info: an optional lgf_pb2.MetaGraphInfo() protobuf
        """
        input_edges = input_edges or []
        output_edges = output_edges or []
        output_node_names = output_node_names or []

        self._nodes = [self._copy_node(node) for node in nodes]
        self._input_edges = []
        input_names = set()
        for edge_info in input_edges:
            tup = (edge_info.name, edge_info.port)
            if tup not in input_names:
                input_names.add(tup)
                self._input_edges.append(self._copy_edge_info(edge_info))

        self._output_edges = [
            self._copy_edge_info(edge_info) for edge_info in output_edges
        ]
        self._output_node_names = list(output_node_names)

        if meta_graph_info is None:
            self._meta_graph_info = lgf_pb2.MetaGraphInfo()
        else:
            self._meta_graph_info = self._copy_meta_graph_info(meta_graph_info)

        # Dictionaries for fast lookups
        self._node_dict = {node.name: node for node in self._nodes}
        self._node_to_input_node_names = {node.name: set() for node in self._nodes}
        self._node_to_output_node_names = {node.name: set() for node in self._nodes}
        self._edge_dict = {}

        for node in self._nodes:
            # Input and output node names
            for e in node.inputs:
                if e.name in self._node_dict:
                    self._node_to_input_node_names[node.name].add(e.name)
                    self._node_to_output_node_names[e.name].add(node.name)
            for inp_name in node.control_inputs:
                if inp_name in self._node_dict:
                    self._node_to_input_node_names[node.name].add(inp_name)
                    self._node_to_output_node_names[inp_name].add(node.name)

            # Edges
            for e in list(node.inputs) + list(node.outputs):
                self._edge_dict[(e.name, e.port)] = e

        # Sort the input and output node names so they are always in the same order
        self._node_to_input_node_names = {
            k: sorted(v) for k,
            v in self._node_to_input_node_names.items()
        }
        self._node_to_output_node_names = {
            k: sorted(v) for k,
            v in self._node_to_output_node_names.items()
        }

        # Make sure required nodes are in the graph
        for node_name in self._meta_graph_info.required_nodes:
            if node_name not in self._node_dict:
                raise ValueError("Required node {} not found in graph".format(node_name))

    def __eq__(self, other_graph):
        node_dict = self.node_dict()
        other_node_dict = other_graph.node_dict()

        if set(node_dict.keys()) != set(other_node_dict.keys()):
            return False

        return all(node_dict[name] == other_node_dict[name] for name in node_dict)

    def _copy_node(self, node):
        node_copy = lgf_pb2.LNF()
        node_copy.CopyFrom(node)
        return node_copy

    def _copy_edge_info(self, edge_info):
        edge_info_copy = lgf_pb2.EdgeInfo()
        edge_info_copy.CopyFrom(edge_info)
        return edge_info_copy

    def _copy_meta_graph_info(self, meta_graph_info):
        meta_graph_info_copy = lgf_pb2.MetaGraphInfo()
        meta_graph_info_copy.CopyFrom(meta_graph_info)
        return meta_graph_info_copy

    def nodes(self):
        """
        Returns a list of nodes in the graph
        Always in the same order as the nodes used to initialize this object
        """
        return [self._copy_node(node) for node in self._nodes]

    def node_dict(self):
        return {node.name: node for node in self.nodes()}

    def get_node_by_name(self, node_name):
        """Returns the node in the graph with the given node_name."""
        return self._copy_node(self._node_dict[node_name])

    def has_node(self, node_name):
        """Returns True if there is a node with the given name"""
        return node_name in self._node_dict

    def get_edge(self, name, port):
        """Returns an edge in the graph with the given name and port"""
        return self._copy_edge_info(self._edge_dict[(name, port)])

    def input_edges(self):
        """
        Returns a list of lgf_pb2.InputInfo() protobufs specifying the inputs of
        the graph. Always in the same order as the inputs used to initialize this object
        """
        return [self._copy_edge_info(edge_info) for edge_info in self._input_edges]

    def output_edges(self):
        """
        Returns a list of lgf_pb2.OutputInfo() protobufs specifying the outputs of
        the graph. Always in the same order as the outputs used to initialize this object
        """
        return [self._copy_edge_info(edge_info) for edge_info in self._output_edges]

    def output_node_names(self):
        """
        Returns a list of strings corresponding to output nodes of the graph
        """
        return list(self._output_node_names)

    def get_input_node_names_of_node(self, node):
        """
        Returns a list of the input node names of the given node
        """
        return list(self._node_to_input_node_names[node.name])

    def get_output_node_names_of_node(self, node):
        """
        Returns a list of the output node names of the given node
        """
        return list(self._node_to_output_node_names[node.name])

    def meta_graph_info(self):
        """
        Returns a lgf_pb2.MetaGraphInfo() protobuf
        """
        return self._copy_meta_graph_info(self._meta_graph_info)

    def prune_graph(self,
                    input_edges=None,
                    output_edges=None,
                    output_node_names=None,
                    include_inputs=True):
        """Returns a new light_graph object."""
        # Inputs and outputs of pruned graph are the same
        input_edges = input_edges or self.input_edges()
        output_edges = output_edges or self.output_edges()
        output_node_names = output_node_names or self.output_node_names()

        # Node filter for input nodes
        input_node_filter = node_filters.and_filter(*[
            node_filters.not_filter(node_filters.name_is_filter(e.name))
            for e in input_edges
        ])

        # Get the root nodes for pruning, include required nodes
        root_nodes = [self.get_node_by_name(e.name) for e in output_edges] + [
            self.get_node_by_name(node_name) for node_name in output_node_names
        ] + [
            self.get_node_by_name(node_name)
            for node_name in self._meta_graph_info.required_nodes
        ]

        # Only keep nodes that the outputs depend on
        nodes = []
        node_names = set()
        for i, root_node in enumerate(root_nodes):
            # Do not use the input node filter for required nodes
            if i < (len(output_edges) + len(output_node_names)):
                node_filter = input_node_filter
            else:
                node_filter = None

            for node in self.bfs(root_node, node_filter=node_filter):
                if node.name not in node_names:
                    nodes.append(node)
                    node_names.add(node.name)

        # Make sure inputs and outputs come from the original graph
        input_edges = [self.get_edge(e.name, e.port) for e in input_edges]
        output_edges = [self.get_edge(e.name, e.port) for e in output_edges]

        # Add input nodes if necessary
        if include_inputs:
            for e in input_edges:
                if e.name in self._node_dict and e.name not in node_names:
                    nodes.append(self._node_dict[e.name])
                    node_names.add(e.name)

        return LightGraph(nodes,
                          input_edges=input_edges,
                          output_edges=output_edges,
                          output_node_names=output_node_names,
                          meta_graph_info=self.meta_graph_info())

    def bfs(self,
            root_node,
            bidirectional=False,
            node_filter=None,
            skip_control_inputs=False):
        """
        Does a BFS on the graph starting at the root_node

        Params:
            root_node: starting node for the BFS
            bidirectional: If False, look at a nodes inputs when doing the BFS and
                discovering new nodes. If True do a bidirectional search, looking at
                a nodes inputs and outputs when discovering new nodes.
            node_filter: If provided, only add nodes to the frontier that match the
                filter with this graph. Note that if the root_node does not match the
                provided filter, no nodes will be returned.
        """
        # Check for unsupported cases
        if bidirectional and skip_control_inputs:
            raise ValueError("Bidirectional BFS is currently unsupported when" +
                             "skipping control inputs")

        # Update node filter with defaults
        default_filter = node_filters.not_filter(
            node_filters.name_starts_with_filter("^"))
        if node_filter is None:
            node_filter = default_filter
        else:
            node_filter = node_filters.and_filter(default_filter, node_filter)

        # Special case when the root_node does not match node_filter
        if not (node_filter.matches(root_node, self)):
            return []

        # BFS
        visited_node_names = {root_node.name}
        current_nodes = [root_node]
        frontier = []
        while current_nodes:
            for parent_node in current_nodes:
                yield self._copy_node(parent_node)

                # Default uses inputs for child nodes
                if skip_control_inputs:
                    # Skip control inputs
                    child_nodes = [
                        self._node_dict[e.name]
                        for e in parent_node.inputs
                        if self.has_node(e.name)
                    ]
                else:
                    # Include control inputs
                    child_nodes = [
                        self._node_dict[n]
                        for n in self._node_to_input_node_names[parent_node.name]
                    ]

                # Bidirectional adds outputs as well, currently always includes
                # control inputs
                if bidirectional:
                    child_nodes += [
                        self._node_dict[n]
                        for n in self._node_to_output_node_names[parent_node.name]
                    ]

                for child_node in child_nodes:
                    if (child_node.name not in visited_node_names
                            and node_filter.matches(child_node,
                                                    self)):
                        visited_node_names.add(child_node.name)
                        frontier.append(child_node)

            current_nodes = frontier
            frontier = []

    @staticmethod
    def _is_const(node):
        if node.HasField(lgf_pb2.LNF.original.DESCRIPTOR.name):
            return node.original.op in LightGraph.CONST_OPS
        else:
            return node.WhichOneof("node") in LightGraph.CONST_NODES

    def is_constant_node(self, node):
        """
        Check whether a node is constant.

        A node is constant provided all of its non-control incoming inputs come from
        constant nodes.
        If a node has no inputs and self._is_const(node) is False, it is defined to be
        a non-constant node.
        If a node is a control flow op, it is defined to be a non-constant node unless
        it is an Enter node with the attribute is_constant == True.
        """
        # Traverse the subtree rooted at node, skipping control inputs
        for child_node in self.bfs(node, skip_control_inputs=True):
            # Control flow ops not constant
            if (child_node.HasField(lgf_pb2.LNF.original.DESCRIPTOR.name)
                    and child_node.original.op in self.CONTROL_FLOW_OPS):
                # Exception for constant enter node
                if child_node.original.op == ops_pb2.ENTER and child_node.original.attr[
                        self.IS_CONST_ATTR].b:
                    continue

                return False

            # Found a leaf of the subtree if
            # 1) The node has no non-control inputs
            # 2) The node has a non-control input edge that does not come from
            #    a node inside the graph (an input edge to the graph)
            if (not len(child_node.inputs)
                    or any([not self.has_node(e.name) for e in child_node.inputs])):
                # Found a non-constant leaf in the subtree
                if not self._is_const(child_node):
                    return False

        return True

    def as_lgf_pb(self):
        """
        Returns the Lightelligence Graph Format (LGF) Protobuf corresponding to
        this graph
        """
        lgf_pb = lgf_pb2.LGF()
        lgf_pb.nodes.extend(self.nodes())
        lgf_pb.input_edges.extend(self.input_edges())
        lgf_pb.output_edges.extend(self.output_edges())
        lgf_pb.output_node_names.extend(self.output_node_names())
        lgf_pb.meta_graph_info.CopyFrom(self.meta_graph_info())

        return lgf_pb

    @classmethod
    def lgf_pb_to_graph(cls, lgf_pb):
        """Converts a LGF Proto to a LightGraph object"""
        return cls(list(lgf_pb.nodes),
                   list(lgf_pb.input_edges),
                   list(lgf_pb.output_edges),
                   list(lgf_pb.output_node_names),
                   meta_graph_info=lgf_pb.meta_graph_info)

    @staticmethod
    def read_lgf_pb(lgf_pb_path):
        """Reads a LGF Proto from the binary file at lgf_pb_path"""
        light_graph = lgf_pb2.LGF()
        with open(lgf_pb_path, "rb") as f:
            light_graph.ParseFromString(f.read())

        return light_graph

    @staticmethod
    def write_lgf_pb(lgf_pb, lgf_pb_path):
        """Writes lgf_pb as a binary file to lgf_pb_path"""
        with open(lgf_pb_path, "wb") as f:
            f.write(lgf_pb.SerializeToString())

    @staticmethod
    def from_pb(lgf_pb_path):
        return LightGraph.lgf_pb_to_graph(LightGraph.read_lgf_pb(lgf_pb_path))


class MutableLightGraph(LightGraph):

    def get_node_by_name(self, node_name):
        """Returns the node in the graph with the given node_name."""
        return self._node_dict[node_name]
