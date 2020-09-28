from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    quantize_transform


class TFLiteQuantizeTransform(quantize_transform.QuantizeTransform):

    def transform(self, quantize_node, light_graph):

        # Get the original sub_node
        sub_node = quantize_node.original

        # Assertions
        self.check_original_node(quantize_node)

        # Get the edges
        input_edge = quantize_node.inputs[0]
        output_edge = quantize_node.outputs[0]

        # Get the quantize attributes
        scale = sub_node.attr["scale"].f
        bias = float(sub_node.attr["zero_point"].i) * scale

        # Return the transforms
        return self.do_generic_transform(quantize_node.name,
                                         input_edge,
                                         output_edge,
                                         quantize_node.control_inputs,
                                         scale,
                                         output_edge.dtype.p,
                                         bias=bias)
