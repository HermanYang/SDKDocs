from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    electronic_op_transform,
)
from lt_sdk.proto import lgf_pb2


class StackTransform(electronic_op_transform.ElectronicOpTransform):

    def create_supported_nodes(self,
                               stack_name,
                               input_edges,
                               output_edge,
                               control_inputs,
                               axis):
        """
        Creates a supported stack node in standard format

        Params:
            stack_name: name of original node
            input_edges: a list of input edges for the original node
            output_edge: output edge for the original node
            control_inputs: a list of node names for the control inputs
            axis: the axis to stack along
        """
        stack_node = self.create_simple_node(stack_name,
                                             lgf_pb2.LNF.stack.DESCRIPTOR.name,
                                             input_edges,
                                             [output_edge],
                                             control_inputs)

        # Attributes
        stack_node.stack.axis = axis

        return [stack_node]

    def can_transform(self, node, light_graph):
        return self.check_single_output(node, light_graph)
