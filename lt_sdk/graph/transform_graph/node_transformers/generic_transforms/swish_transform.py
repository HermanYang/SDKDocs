from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    electronic_op_transform,
    sigmoid_transform,
    vv_transform,
)
from lt_sdk.proto import lgf_pb2


class SwishTransform(electronic_op_transform.ElectronicOpTransform):

    def can_transform(self, node, light_graph):
        return self.check_unary(node, light_graph)

    def transform(self, swish_node, light_graph):
        """
        Converts a swish node to a supported nodes in standard format
        """
        self.check_original_node(swish_node)

        sigmoid_output_edge = lgf_pb2.EdgeInfo()
        sigmoid_output_edge.CopyFrom(swish_node.outputs[0])
        sigmoid_output_edge.name = swish_node.name + "_sigmoid"

        sigmoid_nodes = self.create_transform_obj(
            sigmoid_transform.SigmoidTransform).create_supported_nodes(
                sigmoid_output_edge.name,
                swish_node.inputs[0],
                sigmoid_output_edge,
                swish_node.control_inputs)

        vv_mul_node = self.create_transform_obj(
            vv_transform.VVMulTransform).create_supported_nodes(
                swish_node.name,
                swish_node.inputs[0],
                sigmoid_nodes[0].outputs[0],
                swish_node.outputs[0],
                swish_node.control_inputs)[0]

        return self.create_transform_result(to_add=sigmoid_nodes,
                                            to_replace=[vv_mul_node])
