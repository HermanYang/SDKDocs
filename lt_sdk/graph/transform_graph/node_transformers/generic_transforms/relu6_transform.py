from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    electronic_op_transform,
    sv_transform,
)
from lt_sdk.proto import lgf_pb2


class Relu6Transform(electronic_op_transform.ElectronicOpTransform):
    """Relu6 is in Mobilenet for some reason."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sv_max_transform = sv_transform.SVMaxTransform(*args, **kwargs)
        self._sv_min_transform = sv_transform.SVMinTransform(*args, **kwargs)

    def can_transform(self, node, light_graph):
        return self.check_unary(node, light_graph)

    def transform(self, relu6_node, light_graph):
        """
        Relu6Transform
        """
        self.check_original_node(relu6_node)

        intermediate_edge = lgf_pb2.EdgeInfo()
        intermediate_edge.CopyFrom(relu6_node.outputs[0])
        intermediate_edge.name = relu6_node.name + "_sv_max"

        sv_max_node = self._sv_max_transform.create_supported_nodes(
            intermediate_edge.name,
            relu6_node.inputs[0],
            intermediate_edge,
            relu6_node.control_inputs,
            0)[0]

        sv_min_node = self._sv_min_transform.create_supported_nodes(
            relu6_node.name,
            intermediate_edge,
            relu6_node.outputs[0],
            relu6_node.control_inputs,
            6)[0]

        return self.create_transform_result(to_add=[sv_max_node],
                                            to_replace=[sv_min_node])
