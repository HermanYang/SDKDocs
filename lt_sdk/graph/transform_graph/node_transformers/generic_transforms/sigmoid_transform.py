from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    electronic_op_transform,
    exp_transform,
    sv_transform,
    vv_transform,
)
from lt_sdk.proto import lgf_pb2


class SigmoidTransform(electronic_op_transform.ElectronicOpTransform):

    def create_supported_nodes(self,
                               sigmoid_name,
                               input_edge,
                               output_edge,
                               control_inputs):
        """
        Creates a supported sigmoid node in standard format

        Params:
            sigmoid_name: name of original node
            input_edge: edge of the input for the original node
            output_edge: edge of the output for the original node
            control_inputs: a list of node names for the control inputs
        """
        exp_output_edge = lgf_pb2.EdgeInfo()
        exp_output_edge.CopyFrom(output_edge)
        exp_output_edge.name = sigmoid_name + "_exp"

        exp_node = self.create_transform_obj(
            exp_transform.ExpTransform).create_supported_nodes(
                exp_output_edge.name,
                input_edge,
                exp_output_edge,
                control_inputs)[0]

        sv_add_output_edge = lgf_pb2.EdgeInfo()
        sv_add_output_edge.CopyFrom(output_edge)
        sv_add_output_edge.name = sigmoid_name + "_sv_add"

        sv_add_node = self.create_transform_obj(
            sv_transform.SVAddTransform).create_supported_nodes(
                sv_add_output_edge.name,
                exp_node.outputs[0],
                sv_add_output_edge,
                control_inputs,
                1)[0]

        vv_div_node = self.create_transform_obj(
            vv_transform.VVDivTransform).create_supported_nodes(
                sigmoid_name,
                exp_node.outputs[0],
                sv_add_node.outputs[0],
                output_edge,
                control_inputs)[0]

        return [vv_div_node, sv_add_node, exp_node]

    def can_transform(self, node, light_graph):
        return self.check_unary(node, light_graph)

    def transform(self, sigmoid_node, light_graph):
        self.check_original_node(sigmoid_node)
        return self.do_generic_transform(sigmoid_node.name,
                                         sigmoid_node.inputs[0],
                                         sigmoid_node.outputs[0],
                                         sigmoid_node.control_inputs)
