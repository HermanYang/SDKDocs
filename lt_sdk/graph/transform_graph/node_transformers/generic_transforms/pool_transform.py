from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    electronic_op_transform,
    mean_transform,
)
from lt_sdk.proto import lgf_pb2


class PoolTransform(electronic_op_transform.ElectronicOpTransform):

    def _is_mean(self, input_edge, pool_type, image_attr):
        h_indx, w_indx, c_indx = self._get_height_and_width_indices(image_attr)

        height = input_edge.shape.d[h_indx]
        width = input_edge.shape.d[w_indx]
        kernel_height = image_attr.kernel_size[h_indx]
        kernel_width = image_attr.kernel_size[w_indx]

        return (pool_type == lgf_pb2.PoolNode.AVG_POOL
                and image_attr.padding == lgf_pb2.ImagePatchAttributes.VALID
                and height == kernel_height and width == kernel_width
                and image_attr.kernel_size[0] == 1
                and image_attr.kernel_size[c_indx] == 1)

    def _create_mean_nodes(self,
                           pool_name,
                           input_edge,
                           output_edge,
                           control_inputs,
                           image_attr):
        h_indx, w_indx, _ = self._get_height_and_width_indices(image_attr)
        return self.create_transform_obj(
            mean_transform.MeanTransform).create_supported_nodes(
                pool_name,
                input_edge,
                output_edge,
                control_inputs,
                [h_indx,
                 w_indx],
                True)

    def create_supported_nodes(self,
                               pool_name,
                               input_edge,
                               output_edge,
                               control_inputs,
                               pooling_type,
                               kernel_size,
                               strides,
                               padding,
                               data_format):
        """
        Creates a supported pool node in standard format

        Params:
            pool_name: name of original node
            input_edge: edge of the input for the original node
            output_edge: edge of the output for the original node
            control_inputs: a list of edges for the control inputs
            pooling_type: PoolNode::PoolingType enum
            kernel_size: list or numpy array of 2 or 4 numbers
            strides: list or numpy array of 2 or 4 numbers
            padding: string or enum for padding
            data_format: string or enum for data format
        """
        image_attr = self.create_image_attr(kernel_size, strides, padding, data_format)

        # Special case where average pool can also be a mean
        if self._is_mean(input_edge, pooling_type, image_attr):
            self._create_mean_nodes(pool_name,
                                    input_edge,
                                    output_edge,
                                    control_inputs,
                                    image_attr)

        pool_node = self.create_simple_node(pool_name,
                                            lgf_pb2.LNF.pool.DESCRIPTOR.name,
                                            [input_edge],
                                            [output_edge],
                                            control_inputs)

        # Attributes
        pool_node.pool.pooling_type = pooling_type
        pool_node.pool.image_attr.CopyFrom(image_attr)

        return [pool_node]
