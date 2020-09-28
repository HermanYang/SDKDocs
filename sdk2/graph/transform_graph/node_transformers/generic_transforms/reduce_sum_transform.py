from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    electronic_op_transform
from sdk2.proto import lgf_pb2


class ReduceSumTransform(electronic_op_transform.ElectronicOpTransform):

    def create_supported_nodes(self, reduce_sum_name, input_edge, output_edge,
                               control_inputs, axes, keep_dims):
        """
        Creates a supported reduce sum node in standard format

        Params:
            reduce_sum_name: name of original node
            input_edge: edge of the input for the original node
            output_edge: edge of the output for the original node
            control_inputs: a list of node names for the control inputs
            axes: list or numpy array for which axes to reduce
            keep_dims: boolean, if true keep dimensions of 1
        """
        reduce_sum_node = self.create_simple_node(reduce_sum_name,
                                                  lgf_pb2.LNF.reduce_sum.DESCRIPTOR.name,
                                                  [input_edge], [output_edge],
                                                  control_inputs)

        # Attributes
        reduce_sum_node.reduce_sum.axes.extend(axes)
        reduce_sum_node.reduce_sum.keep_dims = keep_dims

        return [reduce_sum_node]
