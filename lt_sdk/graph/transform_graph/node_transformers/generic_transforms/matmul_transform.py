import logging

from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    opu_op_transform,
)
from lt_sdk.proto import lgf_pb2


class MatMulTransform(opu_op_transform.OPUOpTransform):

    def create_supported_nodes(self,
                               matmul_name,
                               input_edge,
                               weights_edge,
                               output_edge,
                               control_inputs,
                               transpose_inputs=False,
                               transpose_weights=False):
        """
        Creates a supported matmul node in standard format

        Params:
            matmul_name: name of original node
            input_edge: edge of the input for the original node
            weights_edge: input edge for the weights, to be fed to phasify
            output_edge: edge of the output for the original node
            control_inputs: a list of node names for the control inputs
            transpose_inputs: if True, transpose the inputs
            transpose_weights: if True, transpose the weights
        """
        # TODO: support transpose of inputs? just insert a tranpose node?
        if transpose_inputs:
            logging.warning(
                "Transpose of vector in MatMul node {} is NOT supported.".format(
                    matmul_name))
            return []

        # Create a matmul_node
        matmul_node = lgf_pb2.LNF()
        matmul_node.name = matmul_name
        matmul_node.supported = True
        matmul_node.matmul.SetInParent()

        # Phasify
        to_add = self._create_phasify_node(matmul_node,
                                           weights_edge,
                                           transpose_weights=transpose_weights)

        # Input data
        matmul_node.inputs[lgf_pb2.MatMulNode.INPUT_INDEX].CopyFrom(input_edge)
        matmul_node.inputs[lgf_pb2.MatMulNode.INPUT_INDEX].dtype.CopyFrom(
            self._sw_config.float_type)
        matmul_node.control_inputs.extend(control_inputs)

        # Outputs
        matmul_node.outputs.add().CopyFrom(output_edge)
        matmul_node.outputs[0].dtype.CopyFrom(self._sw_config.float_type)

        return [matmul_node] + to_add
