from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    electronic_op_transform,
)
from lt_sdk.proto import lgf_pb2


class ExpTransform(electronic_op_transform.ElectronicOpTransform):

    def create_supported_nodes(self, exp_name, input_edge, output_edge, control_inputs):
        """
        Creates a supported exp node in standard format

        Params:
            exp_name: name of original node
            input_edge: edge of the input for the original node
            output_edge: edge of the output for the original node
            control_inputs: a list of node names for the control inputs
        """
        return [
            self.create_simple_node(exp_name,
                                    lgf_pb2.LNF.exp.DESCRIPTOR.name,
                                    [input_edge],
                                    [output_edge],
                                    control_inputs)
        ]

    def can_transform(self, node, light_graph):
        return self.check_unary(node, light_graph)

    def transform(self, exp_node, light_graph):
        self.check_original_node(exp_node)
        return self.do_generic_transform(exp_node.name,
                                         exp_node.inputs[0],
                                         exp_node.outputs[0],
                                         exp_node.control_inputs)
