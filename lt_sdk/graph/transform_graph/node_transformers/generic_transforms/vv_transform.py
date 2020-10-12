from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    electronic_op_transform,
)
from lt_sdk.proto import lgf_pb2


class VVTransform(electronic_op_transform.ElectronicOpTransform):

    def vv_node_type(self):
        raise NotImplementedError("")

    def create_supported_nodes(self,
                               vv_name,
                               input0_edge,
                               input1_edge,
                               output_edge,
                               control_inputs):
        """
        Creates a supported vv_add node in standard format

        Params:
            vv_name: name of original node
            input0_edge: edge of the first input for the original node
            input1_edge: edge of the second input for the original node
            output_edge: edge of the output for the original node
            control_inputs: a list of node names for the control inputs
        """
        return [
            self.create_simple_node(vv_name,
                                    self.vv_node_type(),
                                    [input0_edge,
                                     input1_edge],
                                    [output_edge],
                                    control_inputs)
        ]

    def can_transform(self, node, light_graph):
        return (len(node.inputs) == 2 and self.check_single_output(node, light_graph))

    def transform(self, vv_node, light_graph):
        self.check_original_node(vv_node)
        return self.do_generic_transform(vv_node.name,
                                         vv_node.inputs[0],
                                         vv_node.inputs[1],
                                         vv_node.outputs[0],
                                         vv_node.control_inputs)


class VVAddTransform(VVTransform):

    def vv_node_type(self):
        return lgf_pb2.LNF.vv_add.DESCRIPTOR.name


class VVMulTransform(VVTransform):

    def vv_node_type(self):
        return lgf_pb2.LNF.vv_mul.DESCRIPTOR.name


class VVDivTransform(VVTransform):

    def vv_node_type(self):
        return lgf_pb2.LNF.vv_div.DESCRIPTOR.name


class VVMaxTransform(VVTransform):

    def vv_node_type(self):
        return lgf_pb2.LNF.vv_max.DESCRIPTOR.name


class VVMinTransform(VVTransform):

    def vv_node_type(self):
        return lgf_pb2.LNF.vv_min.DESCRIPTOR.name


class VVSubTransform(VVTransform):

    def vv_node_type(self):
        return lgf_pb2.LNF.vv_sub.DESCRIPTOR.name
