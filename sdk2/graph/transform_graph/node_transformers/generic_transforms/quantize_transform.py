from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    electronic_op_transform
from sdk2.proto import lgf_pb2


class QuantizeTransform(electronic_op_transform.ElectronicOpTransform):

    def create_supported_nodes(self,
                               quantize_name,
                               input_edge,
                               output_edge,
                               control_inputs,
                               scale,
                               precision,
                               bias=0):
        """
        Creates a supported quant node in standard format

        Params:
            quant_name: name of original node
            input_edge: edge of the input for the original node
            output_edge: edge of the output for the original node
            control_inputs: a list of node names for the control inputs
            scale: quantization scale
            precision: quantization precision
            bias: quantization bias
        """

        # Replace the Quantize Node in order to change the operation to a Quantize
        quant_node = lgf_pb2.LNF()
        quant_node.name = quantize_name
        quant_node.inputs.add().CopyFrom(input_edge)
        quant_node.control_inputs.extend(control_inputs)
        quant_node.outputs.add().CopyFrom(output_edge)
        quant_node.supported = True
        quant_node.quantize.SetInParent()

        # quantize attributes
        quant_node.quantize.scale = scale
        quant_node.quantize.precision = precision
        quant_node.quantize.bias = bias

        return [quant_node]
