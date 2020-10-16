from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    electronic_op_transform,
)
from lt_sdk.proto import lgf_pb2


class TransposeTransform(electronic_op_transform.ElectronicOpTransform):

    def create_supported_nodes(self,
                               transpose_name,
                               input_edge,
                               output_edge,
                               control_inputs,
                               axes):
        """
        Creates a supported transpose node in standard format

        Params:
            transpose_name: name of original node
            input_edge: edge of the input for the original node
            output_edge: edge of the output for the original node
            control_inputs: a list of node names for the control inputs
            axes: list or numpy array for which axes to transpose
        """
        assert (len(input_edge.shape.d) == len(axes))
        assert (len(output_edge.shape.d) == len(axes))

        transpose_node = self.create_simple_node(transpose_name,
                                                 lgf_pb2.LNF.transpose.DESCRIPTOR.name,
                                                 [input_edge],
                                                 [output_edge],
                                                 control_inputs)

        # Attributes
        transpose_node.transpose.axes.extend(axes)

        return [transpose_node]

    def can_transform(self, node, light_graph):
        return self.check_single_output(node, light_graph)
