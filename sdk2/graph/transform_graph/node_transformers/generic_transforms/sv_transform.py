from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    electronic_op_transform
from sdk2.proto import lgf_pb2


class SVTransform(electronic_op_transform.ElectronicOpTransform):

    def sv_node_type(self):
        raise NotImplementedError("")

    def create_supported_nodes(self, sv_name, input_edge, output_edge, control_inputs,
                               scalar):
        """
        Creates a supported sv_* node in standard format

        Params:
            sv_name: name of original node
            input_edge: edge of the input for the original node
            output_edge: edge of the output for the original node
            control_inputs: a list of node names for the control inputs
            scalar: floating point scalar
        """
        sv_node = self.create_simple_node(sv_name, self.sv_node_type(), [input_edge],
                                          [output_edge], control_inputs)

        # Attributes
        getattr(sv_node, self.sv_node_type()).scalar = scalar

        return [sv_node]

    def can_transform(self, node, light_graph):
        return self.check_unary(node, light_graph)


class SVAddTransform(SVTransform):

    def sv_node_type(self):
        return lgf_pb2.LNF.sv_add.DESCRIPTOR.name


class SVMulTransform(SVTransform):

    def sv_node_type(self):
        return lgf_pb2.LNF.sv_mul.DESCRIPTOR.name


class SVMaxTransform(SVTransform):

    def sv_node_type(self):
        return lgf_pb2.LNF.sv_max.DESCRIPTOR.name


class SVMinTransform(SVTransform):

    def sv_node_type(self):
        return lgf_pb2.LNF.sv_min.DESCRIPTOR.name


class SVPowTransform(SVTransform):

    def sv_node_type(self):
        return lgf_pb2.LNF.sv_pow.DESCRIPTOR.name


class ReluTransform(SVMaxTransform):

    def transform(self, relu_node, light_graph):
        """
        Generic ReluTransform
        """
        self.check_original_node(relu_node)

        return self.do_generic_transform(relu_node.name, relu_node.inputs[0],
                                         relu_node.outputs[0], relu_node.control_inputs,
                                         0)


class RsqrtTransform(SVPowTransform):

    def transform(self, rsqrt_node, light_graph):
        """
        Converts original node to a supported transpose in standard format
        """
        self.check_original_node(rsqrt_node)

        return self.do_generic_transform(rsqrt_node.name, rsqrt_node.inputs[0],
                                         rsqrt_node.outputs[0],
                                         rsqrt_node.control_inputs, -0.5)
