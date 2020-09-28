import numpy as np

from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    electronic_op_transform
from sdk2.proto import lgf_pb2


class PadTransform(electronic_op_transform.ElectronicOpTransform):

    def create_supported_nodes(self, pad_name, input_edge, output_edge, control_inputs,
                               padding):
        """
        Creates a supported pad node in standard format

        Params:
            pad_name: name of original node
            input_edge: edge of the input for the original node
            output_edge: edge of the output for the original node
            control_inputs: a list of node names for the control inputs
            padding: a 2D list or numpy array of (before, after) pairs for each dim
        """
        pad_node = self.create_simple_node(pad_name, lgf_pb2.LNF.pad.DESCRIPTOR.name,
                                           [input_edge], [output_edge], control_inputs)

        # Attributes
        pad_node.pad.padding.extend(np.array(padding).flatten())

        return [pad_node]
