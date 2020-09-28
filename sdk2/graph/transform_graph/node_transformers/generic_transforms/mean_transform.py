from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    electronic_op_transform
from sdk2.proto import lgf_pb2


class MeanTransform(electronic_op_transform.ElectronicOpTransform):

    def create_supported_nodes(self, mean_name, input_edge, output_edge, control_inputs,
                               axes, keep_dims):
        """
        Creates a supported mean node in standard format

        Params:
            mean_name: name of original node
            input_edge: edge of the input for the original node
            output_edge: edge of the output for the original node
            control_inputs: a list of node names for the control inputs
            axes: list or numpy array for which axes to reduce
            keep_dims: boolean, if true keep dimensions of 1
        """
        mean_node = self.create_simple_node(mean_name, lgf_pb2.LNF.mean.DESCRIPTOR.name,
                                            [input_edge], [output_edge], control_inputs)

        # Attributes
        mean_node.mean.axes.extend(axes)
        mean_node.mean.keep_dims = keep_dims

        return [mean_node]
