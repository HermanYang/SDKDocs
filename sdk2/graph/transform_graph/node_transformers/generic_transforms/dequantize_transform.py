import numpy as np

from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    electronic_op_transform
from sdk2.proto import lgf_pb2


class DequantizeTransform(electronic_op_transform.ElectronicOpTransform):

    def create_supported_nodes(self,
                               dequant_name,
                               input_edge,
                               output_edge,
                               control_inputs,
                               scales,
                               bias=0,
                               method=lgf_pb2.DQ_STANDARD):
        """
        Creates a supported dequant node in standard format

        Params:
            dequant_name: name of original node
            input_edge: edge of the input for the original node
            output_edge: edge of the output for the original node
            control_inputs: a list of node names for the control inputs
            scales: a list or numpy array of dequant scales
            bias: dequant bias
            method: a lgf_pb2.DequantMethod
        """

        # Create the dequant scales const node
        dequant_scales_node = self.create_const_node(np.array(scales),
                                                     dequant_name + "_scales",
                                                     self._sw_config.float_type,
                                                     lgf_pb2.ConstNode.DEQUANT_SCALE)

        # Create dequant node
        dequant_node = lgf_pb2.LNF()
        dequant_node.name = dequant_name
        dequant_node.inputs.add().CopyFrom(input_edge)
        dequant_node.inputs.add().CopyFrom(dequant_scales_node.outputs[0])
        dequant_node.control_inputs.extend(control_inputs)
        dequant_node.outputs.add().CopyFrom(output_edge)
        dequant_node.supported = True
        dequant_node.dequantize.SetInParent()

        # dequantize attributes
        dequant_node.dequantize.method = method
        dequant_node.dequantize.bias = bias

        return [dequant_node, dequant_scales_node]
