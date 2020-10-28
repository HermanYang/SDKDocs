from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    electronic_op_transform,
)
from lt_sdk.proto import lgf_pb2


class ConcatTransform(electronic_op_transform.ElectronicOpTransform):

    def create_supported_nodes(self,
                               concat_name,
                               input_edges,
                               output_edge,
                               control_inputs,
                               axis):
        """
        Creates a supported concat node in standard format

        Params:
            concat_name: name of original node
            input_edges: a list of input edges for the original node
            output_edge: output edge for the original node
            control_inputs: a list of node names for the control inputs
            axis: the axis to concatenate along
        """
        concat_node = self.create_simple_node(concat_name,
                                              lgf_pb2.LNF.concat.DESCRIPTOR.name,
                                              input_edges,
                                              [output_edge],
                                              control_inputs)

        # Attributes
        concat_node.concat.axis = axis

        return [concat_node]

    def can_transform(self, node, light_graph):
        return self.check_single_output(node, light_graph)
