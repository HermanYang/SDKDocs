from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    electronic_op_transform,
    exp_transform,
    sv_transform,
    vv_transform,
)
from lt_sdk.proto import lgf_pb2


class TanhTransform(electronic_op_transform.ElectronicOpTransform):

    def create_supported_nodes(self, tanh_name, input_edge, output_edge, control_inputs):
        """
        Creates a supported tanh node in standard format

        Params:
            tanh_name: name of original node
            input_edge: edge of the input for the original node
            output_edge: edge of the output for the original node
            control_inputs: a list of node names for the control inputs
        """
        sv_mul_output_edge = lgf_pb2.EdgeInfo()
        sv_mul_output_edge.CopyFrom(output_edge)
        sv_mul_output_edge.name = tanh_name + "_double"

        sv_mul_node = self.create_transform_obj(
            sv_transform.SVMulTransform).create_supported_nodes(
                sv_mul_output_edge.name,
                input_edge,
                sv_mul_output_edge,
                control_inputs,
                2)[0]

        exp_output_edge = lgf_pb2.EdgeInfo()
        exp_output_edge.CopyFrom(output_edge)
        exp_output_edge.name = tanh_name + "_double_exp"

        exp_node = self.create_transform_obj(
            exp_transform.ExpTransform).create_supported_nodes(
                exp_output_edge.name,
                sv_mul_node.outputs[0],
                exp_output_edge,
                control_inputs)[0]

        numerator_output_edge = lgf_pb2.EdgeInfo()
        numerator_output_edge.CopyFrom(output_edge)
        numerator_output_edge.name = tanh_name + "_numerator"
        denominator_output_edge = lgf_pb2.EdgeInfo()
        denominator_output_edge.CopyFrom(output_edge)
        denominator_output_edge.name = tanh_name + "_denominator"

        numerator_node = self.create_transform_obj(
            sv_transform.SVAddTransform).create_supported_nodes(
                numerator_output_edge.name,
                exp_node.outputs[0],
                numerator_output_edge,
                control_inputs,
                -1)[0]
        denominator_node = self.create_transform_obj(
            sv_transform.SVAddTransform).create_supported_nodes(
                denominator_output_edge.name,
                exp_node.outputs[0],
                denominator_output_edge,
                control_inputs,
                1)[0]

        vv_div_node = self.create_transform_obj(
            vv_transform.VVDivTransform).create_supported_nodes(
                tanh_name,
                numerator_node.outputs[0],
                denominator_node.outputs[0],
                output_edge,
                control_inputs)[0]

        return [vv_div_node, numerator_node, denominator_node, exp_node, sv_mul_node]

    def can_transform(self, node, light_graph):
        return self.check_unary(node, light_graph)

    def transform(self, tanh_node, light_graph):
        self.check_original_node(tanh_node)
        return self.do_generic_transform(tanh_node.name,
                                         tanh_node.inputs[0],
                                         tanh_node.outputs[0],
                                         tanh_node.control_inputs)
