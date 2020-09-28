import numpy as np

from sdk2.graph.transform_graph.graph_transformers import apply_node_map
from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    base_transform
from sdk2.proto import lgf_pb2, node_filters


class RemovePadNodeTransform(base_transform.BaseTransform):
    """
    Converts --> pad --> conv2d (VALID) -->
    to --> conv2d (SAME)--> if possible

    Equations are from stack overflow
    https://stackoverflow.com/a/44242277
    """

    @staticmethod
    def _get_same_output_height_and_width(input_height, input_width, stride_height,
                                          stride_width):
        output_height = np.ceil(input_height / stride_height).astype(int)
        output_width = np.ceil(input_width / stride_width).astype(int)
        return output_height, output_width

    @staticmethod
    def _get_same_padding(input_height, input_width, output_height, output_width,
                          kernel_height, kernel_width, stride_height, stride_width):
        pad_along_height = max(
            (output_height - 1) * stride_height + kernel_height - input_height, 0)
        pad_along_width = max(
            (output_width - 1) * stride_width + kernel_width - input_width, 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        return pad_top, pad_bottom, pad_left, pad_right

    def _get_converted_pad_type(self, pad_node, conv2d_node):
        """
        Returns pad type to remove pad_node followed by conv2d_node, returns
        None if removal is not possible
        """

        # Initialize some things
        image_attr = conv2d_node.conv2d.image_attr
        assert (image_attr.padding == lgf_pb2.ImagePatchAttributes.VALID)
        h_indx, w_indx, c_indx = self._get_height_and_width_indices(image_attr)
        kernel_size = image_attr.kernel_size
        strides = image_attr.strides

        # Padding from pad node cannot pad batch or channel
        padding = pad_node.pad.padding
        if (padding[0] != 0 or padding[1] != 0 or padding[2 * c_indx] != 0
                or padding[2 * c_indx + 1] != 0):
            return None

        # Get pad input shape and conv2d output shape
        input_shape = pad_node.inputs[0].shape.d
        output_shape = conv2d_node.outputs[0].shape.d

        # Get the output shape that a SAME conv2d would create
        same_output_height, same_output_width = self._get_same_output_height_and_width(
            input_shape[h_indx], input_shape[w_indx], strides[h_indx], strides[w_indx])

        # Check to see that actual output matches same output
        if (output_shape[h_indx] != same_output_height
                or output_shape[w_indx] != same_output_width):
            return None

        # Get the padding that a SAME conv2d would add
        pad_top, pad_bottom, pad_left, pad_right = self._get_same_padding(
            input_shape[h_indx], input_shape[w_indx], output_shape[h_indx],
            output_shape[w_indx], kernel_size[h_indx], kernel_size[w_indx],
            strides[h_indx], strides[w_indx])

        # Check if padding matches one of our padding schemes
        if (pad_top == padding[2 * h_indx] and pad_bottom == padding[2 * h_indx + 1]
                and pad_left == padding[2 * w_indx]
                and pad_right == padding[2 * w_indx + 1]):
            return lgf_pb2.ImagePatchAttributes.SAME
        elif (pad_bottom == padding[2 * h_indx] and pad_bottom == padding[2 * h_indx + 1]
              and pad_right == padding[2 * w_indx]
              and pad_right == padding[2 * w_indx + 1]):
            return lgf_pb2.ImagePatchAttributes.SAME_EVEN
        else:
            return None

    def _can_be_converted_to_same(self, pad_node, conv2d_node):
        return self._get_converted_pad_type(pad_node, conv2d_node) is not None

    def can_transform(self, pad_node, light_graph):
        # Pad node must have one input
        if len(pad_node.inputs) > 1:
            return False

        for node_name in light_graph.get_output_node_names_of_node(pad_node):
            node = light_graph.get_node_by_name(node_name)

            # All outputs must be conv2d
            if not node.HasField(lgf_pb2.LNF.conv2d.DESCRIPTOR.name):
                return False

            # All conv2d must have VALID padding
            if node.conv2d.image_attr.padding != lgf_pb2.ImagePatchAttributes.VALID:
                return False

            # Padding from pad node must match pad that SAME conv2d will apply
            if not self._can_be_converted_to_same(pad_node, node):
                return False

        return True

    def transform(self, pad_node, light_graph):
        to_replace = []

        for node_name in light_graph.get_output_node_names_of_node(pad_node):
            conv2d_node = light_graph.get_node_by_name(node_name)

            new_conv2d_node = lgf_pb2.LNF()
            new_conv2d_node.CopyFrom(conv2d_node)
            new_conv2d_node.conv2d.image_attr.padding = self._get_converted_pad_type(
                pad_node, conv2d_node)
            new_conv2d_node.inputs[lgf_pb2.MatMulNode.INPUT_INDEX].CopyFrom(
                pad_node.inputs[0])

            to_replace.append(new_conv2d_node)

        return self.create_transform_result(to_replace=to_replace)


class RemovePadNodes(apply_node_map.ApplyNodeMap):

    def __init__(self, hw_specs, sw_config, sim_params):
        super().__init__(
            hw_specs, sw_config, {
                node_filters.which_oneof_filter(lgf_pb2.LNF.pad.DESCRIPTOR.name):
                    RemovePadNodeTransform(hw_specs, sw_config, sim_params)
            })
