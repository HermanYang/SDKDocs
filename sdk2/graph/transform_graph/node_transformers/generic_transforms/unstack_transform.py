from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    electronic_op_transform
from sdk2.proto import lgf_pb2


class UnstackTransform(electronic_op_transform.ElectronicOpTransform):

    def create_supported_nodes(self, unstack_name, input_edge, output_edges,
                               control_inputs, axis):
        """
        Creates a supported unstack node in standard format

        Params:
            unstack_name: name of original node
            input_edge: edge of the input for the original node
            output_edges: a list of output edges for the original node
            control_inputs: a list of node names for the control inputs
            axis: the axis to unstack along
        """
        unstack_node = self.create_simple_node(unstack_name,
                                               lgf_pb2.LNF.unstack.DESCRIPTOR.name,
                                               [input_edge], output_edges,
                                               control_inputs)

        # Attributes
        unstack_node.unstack.axis = axis

        return [unstack_node]

    def can_transform(self, node, light_graph):
        return len(node.inputs) == 1 and super().can_transform(node, light_graph)
