import numpy as np

from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    electronic_op_transform
from sdk2.proto import lgf_pb2


class BatchNormTransform(electronic_op_transform.ElectronicOpTransform):

    NUM_INPUTS = len(lgf_pb2.FusedBatchNormNode.index.DESCRIPTOR.values)

    def create_supported_nodes(self, bn_name, input_edge, output_edge, control_inputs,
                               mean, variance, scale, bias, epsilon):
        """
        Creates a supported batchnorm node in standard format

        Params:
            bn_name: name of original node
            input_edge: edge of the input for the original node
            output_edge: edge of the output for the original node
            control_inputs: a list of node names for the control inputs
            mean: list or numpy array for the mean
            variance: list of numpy array for the variance
            scale: list or numpy array for the scale
            bias: list or numpy array for the scale
            epsilon: float for epsilon
        """
        # Create constant nodes
        mean_node = self.create_const_node(
            np.array(mean).flatten(), bn_name + "_mean", self._sw_config.float_type,
            lgf_pb2.ConstNode.GRAPH_CONST)
        variance_node = self.create_const_node(
            np.array(variance).flatten(), bn_name + "_variance",
            self._sw_config.float_type, lgf_pb2.ConstNode.GRAPH_CONST)
        scale_node = self.create_const_node(
            np.array(scale).flatten(), bn_name + "_scale", self._sw_config.float_type,
            lgf_pb2.ConstNode.GRAPH_CONST)
        bias_node = self.create_const_node(
            np.array(bias).flatten(), bn_name + "_bias", self._sw_config.float_type,
            lgf_pb2.ConstNode.GRAPH_CONST)

        # Create list of input edges
        inputs = [None] * self.NUM_INPUTS
        inputs[lgf_pb2.FusedBatchNormNode.INPUT_INDEX] = input_edge
        inputs[lgf_pb2.FusedBatchNormNode.MEAN_INDEX] = mean_node.outputs[0]
        inputs[lgf_pb2.FusedBatchNormNode.VARIANCE_INDEX] = variance_node.outputs[0]
        inputs[lgf_pb2.FusedBatchNormNode.SCALE_INDEX] = scale_node.outputs[0]
        inputs[lgf_pb2.FusedBatchNormNode.BIAS_INDEX] = bias_node.outputs[0]

        # Create batch norm node
        bn_node = self.create_simple_node(bn_name, lgf_pb2.LNF.batchnorm.DESCRIPTOR.name,
                                          inputs, [output_edge], control_inputs)
        bn_node.batchnorm.epsilon = epsilon

        return [bn_node, mean_node, variance_node, scale_node, bias_node]
