from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    sv_transform,
    vv_transform,
)
from lt_sdk.graph.transform_graph.node_transformers.tf_transforms import (
    tf_base_transform,
)
from lt_sdk.proto import lgf_pb2


class TFSavedModelSquaredDifferenceTransform(tf_base_transform.TFSavedModelBaseTransform,
                                             vv_transform.VVTransform):

    def create_supported_nodes(self,
                               sq_diff_name,
                               input0_edge,
                               input1_edge,
                               output_edge,
                               control_inputs):
        """
        Creates a supported tanh node in standard format

        Params:
            sq_diff_name: name of original node
            input0_edge: edge of the first input for the original node
            input1_edge: edge of the second input for the original node
            output_edge: edge of the output for the original node
            control_inputs: a list of node names for the control inputs
        """

        common_args = self._common_args()

        vv_sub_output_edge = lgf_pb2.EdgeInfo()
        vv_sub_output_edge.CopyFrom(output_edge)
        vv_sub_output_edge.name = sq_diff_name + "_sub"

        vv_sub_node = vv_transform.VVSubTransform(*common_args).create_supported_nodes(
            vv_sub_output_edge.name,
            input0_edge,
            input1_edge,
            vv_sub_output_edge,
            control_inputs)[0]

        sv_pow_node = sv_transform.SVPowTransform(*common_args).create_supported_nodes(
            sq_diff_name,
            vv_sub_node.outputs[0],
            output_edge,
            control_inputs,
            2)[0]

        return [sv_pow_node, vv_sub_node]

    def transform(self, sq_diff_node, light_graph):
        self.check_original_node(sq_diff_node, graph_type=self.GRAPH_TYPE)
        return self.do_generic_transform(sq_diff_node.name,
                                         sq_diff_node.inputs[0],
                                         sq_diff_node.inputs[1],
                                         sq_diff_node.outputs[0],
                                         sq_diff_node.control_inputs)
