from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    electronic_op_transform,
)
from lt_sdk.proto import lgf_pb2


class ReshapeTransform(electronic_op_transform.ElectronicOpTransform):

    def create_supported_nodes(self,
                               reshape_name,
                               input_edge,
                               output_edge,
                               control_inputs):
        """
        Creates a supported reshape node in standard format

        Params:
            reshape_name: name of original node
            input_edge: edge of the input for the original node
            output_edge: edge of the output for the original node
            control_inputs: a list of node names for the control inputs
        """
        return [
            self.create_simple_node(reshape_name,
                                    lgf_pb2.LNF.reshape.DESCRIPTOR.name,
                                    [input_edge],
                                    [output_edge],
                                    control_inputs)
        ]

    def can_transform(self, node, light_graph):
        return self.check_single_output(node, light_graph)

    def transform(self, reshape_node, light_graph):
        """
        Generic ReshapeTransform
        """
        # TODO Reshape that uses dynamic tensor for the new shape?
        # Currently assumes the new shape is a constant tensor and ignores that input
        self.check_original_node(reshape_node)
        return self.do_generic_transform(reshape_node.name,
                                         reshape_node.inputs[0],
                                         reshape_node.outputs[0],
                                         reshape_node.control_inputs)
