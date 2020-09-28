from sdk2.proto import lgf_pb2, node_filter_pb2


class NodeFilter(object):
    """A wrapper implementing functionality for the node_filter protobuf"""

    def __init__(self, node_filter_proto):
        # Get the proto for the specific filter
        self._filter_type = node_filter_proto.WhichOneof("node_filter")
        self._proto = getattr(node_filter_proto, self._filter_type)

        # Build recursive filters in constructor
        if hasattr(self._proto, "filters"):
            self._filters = [NodeFilter(f) for f in self._proto.filters]
        if hasattr(self._proto, "filter"):
            self._filter = NodeFilter(self._proto.filter)

        # Every field name maps to a private function that implements matches
        self._matches_map = {
            filter_name: getattr(self, "_{}".format(filter_name)) for filter_name in
            node_filter_pb2.NodeFilter.DESCRIPTOR.fields_by_name.keys()
        }

    def _true_filter(self, node, light_graph):
        return True

    def _and_filter(self, node, light_graph):
        return all(f.matches(node, light_graph) for f in self._filters)

    def _or_filter(self, node, light_graph):
        return any(f.matches(node, light_graph) for f in self._filters)

    def _not_filter(self, node, light_graph):
        return not (self._filter.matches(node, light_graph))

    def _name_is_filter(self, node, light_graph):
        return self._proto.name == node.name

    def _name_in_filter(self, node, light_graph):
        return self._proto.name in node.name

    def _name_starts_with_filter(self, node, light_graph):
        return node.name.startswith(self._proto.prefix)

    def _first_node_filter(self, node, light_graph):
        return node.name in [e.name for e in light_graph.inputs()]

    def _last_node_filter(self, node, light_graph):
        return node.name in [e.name for e in light_graph.outputs()]

    def _op_filter(self, node, light_graph):
        assert (node.HasField(lgf_pb2.LNF.original.DESCRIPTOR.name))
        return node.original.op in self._proto.ops

    def _supported_node_filter(self, node, light_graph):
        return node.supported

    def _which_oneof_filter(self, node, light_graph):
        return node.WhichOneof("node") in self._proto.oneofs

    def matches(self, node, light_graph):
        """Return true if the given node matches this filter.

        params:
            node: A LightNode object
            light_graph: A LightGraph object
        returns: True if node matches, False otherwise.
        """
        return self._matches_map[self._filter_type](node, light_graph)

    def as_proto(self):
        node_filter_proto = node_filter_pb2.NodeFilter()
        getattr(node_filter_proto, self._filter_type).SetInParent()
        getattr(node_filter_proto, self._filter_type).CopyFrom(self._proto)
        return node_filter_proto


# ----------------- Helper methods to construct NodeFilter objects ----------------------


def true_filter():
    node_filter_proto = node_filter_pb2.NodeFilter()
    node_filter_proto.true_filter.SetInParent()

    return NodeFilter(node_filter_proto)


def false_filter():
    return not_filter(true_filter())


def and_filter(*filters):
    node_filter_proto = node_filter_pb2.NodeFilter()
    node_filter_proto.and_filter.SetInParent()
    node_filter_proto.and_filter.filters.extend([f.as_proto() for f in filters])

    return NodeFilter(node_filter_proto)


def or_filter(*filters):
    node_filter_proto = node_filter_pb2.NodeFilter()
    node_filter_proto.or_filter.SetInParent()
    node_filter_proto.or_filter.filters.extend([f.as_proto() for f in filters])

    return NodeFilter(node_filter_proto)


def not_filter(filter):
    node_filter_proto = node_filter_pb2.NodeFilter()
    node_filter_proto.not_filter.SetInParent()
    node_filter_proto.not_filter.filter.CopyFrom(filter.as_proto())

    return NodeFilter(node_filter_proto)


def name_is_filter(name):
    node_filter_proto = node_filter_pb2.NodeFilter()
    node_filter_proto.name_is_filter.SetInParent()
    node_filter_proto.name_is_filter.name = name

    return NodeFilter(node_filter_proto)


def name_in_filter(name):
    node_filter_proto = node_filter_pb2.NodeFilter()
    node_filter_proto.name_in_filter.SetInParent()
    node_filter_proto.name_in_filter.name = name

    return NodeFilter(node_filter_proto)


def name_starts_with_filter(prefix):
    node_filter_proto = node_filter_pb2.NodeFilter()
    node_filter_proto.name_starts_with_filter.SetInParent()
    node_filter_proto.name_starts_with_filter.prefix = prefix

    return NodeFilter(node_filter_proto)


def first_node_filter():
    node_filter_proto = node_filter_pb2.NodeFilter()
    node_filter_proto.first_node_filter.SetInParent()

    return NodeFilter(node_filter_proto)


def last_node_filter():
    node_filter_proto = node_filter_pb2.NodeFilter()
    node_filter_proto.last_node_filter.SetInParent()

    return NodeFilter(node_filter_proto)


def op_filter(*ops):
    node_filter_proto = node_filter_pb2.NodeFilter()
    node_filter_proto.op_filter.SetInParent()
    node_filter_proto.op_filter.ops.extend(ops)

    return NodeFilter(node_filter_proto)


def supported_node_filter():
    node_filter_proto = node_filter_pb2.NodeFilter()
    node_filter_proto.supported_node_filter.SetInParent()

    return NodeFilter(node_filter_proto)


def which_oneof_filter(*oneofs):
    node_filter_proto = node_filter_pb2.NodeFilter()
    node_filter_proto.which_oneof_filter.SetInParent()
    node_filter_proto.which_oneof_filter.oneofs.extend(oneofs)

    return NodeFilter(node_filter_proto)


def prefixes_filter(prefixes):
    node_filter = false_filter()

    for prefix in prefixes:
        node_filter = or_filter(name_starts_with_filter(prefix), node_filter)

    return node_filter


# ---------------------------------------------------------------------------------------
