from sdk2.graph.transform_graph.node_transformers.generic_transforms import (
    conv2d_transform, depthwise_conv2d_transform)
from sdk2.graph.transform_graph.node_transformers.tf_transforms import tf_base_transform


class TFSavedModelConv2DTransformBase(tf_base_transform.TFSavedModelBaseTransform):

    def transform(self, conv2d_node, light_graph):
        """
        Converts original node to a supported conv2d in standard format
        """
        self.check_original_node(conv2d_node, graph_type=self.GRAPH_TYPE)
        tf_attr = self._get_tf_attr(conv2d_node)

        # Get input and weight nodes
        input_node, weight_node = self.find_input_and_weight_nodes(
            conv2d_node, light_graph)
        input_index = self._get_input_index(input_node, conv2d_node)

        # Get kernel size
        kernel_size = [
            weight_node.outputs[0].shape.d[0], weight_node.outputs[0].shape.d[1]
        ]

        # Get strides
        strides = list(tf_attr["strides"].list.i)
        if len(strides) == 1:
            strides += strides

        # Padding and data format
        padding = self._get_string(tf_attr, "padding")
        data_format = self._get_string(tf_attr, "data_format")

        return self.do_generic_transform(conv2d_node.name,
                                         conv2d_node.inputs[input_index],
                                         weight_node.outputs[0], conv2d_node.outputs[0],
                                         conv2d_node.control_inputs, kernel_size,
                                         strides, padding, data_format)


class TFSavedModelConv2DTransform(TFSavedModelConv2DTransformBase,
                                  conv2d_transform.Conv2DTransform):
    pass


class TFSavedModelDepthwiseConv2DTransform(
        TFSavedModelConv2DTransformBase,
        depthwise_conv2d_transform.DepthwiseConv2DTransform):
    pass
