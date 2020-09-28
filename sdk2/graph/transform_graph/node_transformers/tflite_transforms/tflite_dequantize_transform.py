import numpy as np

from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    dequantize_transform


class TFLiteDequantizeTransform(dequantize_transform.DequantizeTransform):

    def transform(self, dequantize_node, light_graph):

        # Get the original sub_node
        sub_node = dequantize_node.original

        # Assertions
        self.check_original_node(dequantize_node)

        # Get the edges
        input_edge = dequantize_node.inputs[0]
        output_edge = dequantize_node.outputs[0]

        # Get the dequantize attributes
        scales = np.array([sub_node.attr["scale"].f])
        assert (len(scales) == 1)  # Only one scale is supported for now
        bias = float(sub_node.attr["zero_point"].i) * scales[0]

        # Return the transforms
        return self.do_generic_transform(dequantize_node.name, input_edge, output_edge,
                                         dequantize_node.control_inputs, scales, bias)
