from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    electronic_op_transform,
)
from lt_sdk.proto import lgf_pb2


class IdentityTransform(electronic_op_transform.ElectronicOpTransform):

    def create_supported_nodes(self,
                               identity_name,
                               input_edge,
                               output_edge,
                               control_inputs):
        """
        Creates a supported identity node in standard format

        Params:
            identity_name: name of original node
            input_edge: edge of the input for the original node
            output_edge: edge of the output for the original node
            control_inputs: a list of node names for the control inputs
        """
        return [
            self.create_simple_node(identity_name,
                                    lgf_pb2.LNF.identity.DESCRIPTOR.name,
                                    [input_edge],
                                    [output_edge],
                                    control_inputs)
        ]

    def can_transform(self, node, light_graph):
        return self.check_unary(node, light_graph)

    def transform(self, identity_node, light_graph):
        """
        Generic IdentityTransform
        """
        self.check_original_node(identity_node)

        return self.do_generic_transform(identity_node.name,
                                         identity_node.inputs[0],
                                         identity_node.outputs[0],
                                         identity_node.control_inputs)
