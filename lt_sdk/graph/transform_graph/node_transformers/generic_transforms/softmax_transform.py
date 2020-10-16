from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    exp_transform,
    reduce_sum_transform,
    vv_transform,
)
from lt_sdk.proto import lgf_pb2


class SoftmaxTransform(exp_transform.ExpTransform,
                       reduce_sum_transform.ReduceSumTransform,
                       vv_transform.VVDivTransform):

    def create_supported_nodes(self,
                               softmax_name,
                               input_edge,
                               output_edge,
                               control_inputs,
                               axis):
        """
        Creates a supported softmax node in standard format

        Params:
            softmax_name: name of original node
            input_edge: edge of the input for the original node
            output_edge: edge of the output for the original node
            control_inputs: a list of node names for the control inputs
            axes: integer for which dimension to do softmax over
        """
        exp_output_edge = lgf_pb2.EdgeInfo()
        exp_output_edge.CopyFrom(output_edge)
        exp_output_edge.name = softmax_name + "_exp"

        exp_node = exp_transform.ExpTransform.create_supported_nodes(
            self,
            exp_output_edge.name,
            input_edge,
            exp_output_edge,
            control_inputs)[0]

        reduce_sum_output_edge = lgf_pb2.EdgeInfo()
        reduce_sum_output_edge.CopyFrom(output_edge)
        reduce_sum_output_edge.name = softmax_name + "_reduce_sum"
        reduce_sum_output_edge.shape.d[axis] = 1

        reduce_sum_node = reduce_sum_transform.ReduceSumTransform.create_supported_nodes(
            self,
            reduce_sum_output_edge.name,
            exp_node.outputs[0],
            reduce_sum_output_edge,
            control_inputs,
            [axis],
            True)[0]

        vv_div_node = vv_transform.VVDivTransform.create_supported_nodes(
            self,
            softmax_name,
            exp_node.outputs[0],
            reduce_sum_node.outputs[0],
            output_edge,
            control_inputs)[0]

        return [vv_div_node, reduce_sum_node, exp_node]
