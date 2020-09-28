from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    electronic_op_transform
from sdk2.proto import lgf_pb2


class SplitTransform(electronic_op_transform.ElectronicOpTransform):

    def create_supported_nodes(self, split_name, input_edge, output_edges,
                               control_inputs, axis):
        """
        Creates a supported split node in standard format

        Params:
            split_name: name of original node
            input_edge: input edge for the original node
            output_edges: a list of output edges for the original node
            control_inputs: a list of node names for the control inputs
            axis: the axis to split along
        """
        split_node = self.create_simple_node(split_name,
                                             lgf_pb2.LNF.split.DESCRIPTOR.name,
                                             [input_edge], output_edges, control_inputs)

        # Attributes
        split_node.split.axis = axis

        return [split_node]
