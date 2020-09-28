from sdk2.graph.transform_graph.node_transformers.generic_transforms import (
    electronic_op_transform, matmul_transform)
from sdk2.proto import lgf_pb2, transform_result_pb2


class TFLiteFullyConnectedTransform(matmul_transform.MatMulTransform,
                                    electronic_op_transform.ElectronicOpTransform):

    def __init__(self, *args, **kwargs):
        self._dtype_override = None
        super().__init__(*args, **kwargs)

    @property
    def _dequantize_before_accumulate(self):
        return True

    def _get_dtype(self):
        if self._dtype_override:
            return self._dtype_override
        else:
            return super()._get_dtype()

    def transform(self, fully_connected_node, light_graph):
        """
        Converts unsupported TFLite FULLY_CONNECTED to supported standard MATMUL
        and VV_ADD
        """
        # Get the original sub_node
        sub_node = fully_connected_node.original

        # Assertions
        self.check_original_node(fully_connected_node)
        assert ("fused_activation_function" in sub_node.attr)
        assert ("weights_format" in sub_node.attr)
        assert (sub_node.attr["weights_format"].s == "DEFAULT")

        # Get the edges
        input_edge = fully_connected_node.inputs[0]
        weight_edge = fully_connected_node.inputs[1]
        bias_edge = fully_connected_node.inputs[2]
        output_edge = fully_connected_node.outputs[0]

        # Transforms
        to_add = []
        to_replace = []
        to_reroute = []
        edge_reroute = transform_result_pb2.ToReroute.edge_reroute.DESCRIPTOR.name

        # Matmul transform
        matmul_transforms = matmul_transform.MatMulTransform.do_generic_transform(
            self, fully_connected_node.name, input_edge, weight_edge, output_edge,
            fully_connected_node.control_inputs)
        to_add.extend([t.node for t in matmul_transforms.to_add])
        to_replace.extend([t.node for t in matmul_transforms.to_replace])
        matmul_node = matmul_transforms.to_replace[0].node

        # Create the CAST node to cast the bias to the type of the matmul
        cast_bias_node = lgf_pb2.LNF()
        cast_bias_node.name = fully_connected_node.name + "_cast_bias"
        cast_bias_node.supported = True
        cast_bias_node.cast.SetInParent()
        # Create the CAST input edge
        cast_bias_input_edge = cast_bias_node.inputs.add()
        cast_bias_input_edge.CopyFrom(bias_edge)
        cast_bias_input_edge.dtype.CopyFrom(self._get_dtype())
        # Create the CAST output edge
        cast_bias_edge = cast_bias_node.outputs.add()
        cast_bias_edge.name = cast_bias_node.name
        cast_bias_edge.port = 0
        cast_bias_edge.dtype.CopyFrom(matmul_node.outputs[0].dtype)
        cast_bias_edge.shape.CopyFrom(bias_edge.shape)
        # Append the node to add
        to_add.append(cast_bias_node)
        # Append the edges to be rerouted: BIAS>FC to BIAS>CAST_BIAS
        to_reroute.extend([(edge_reroute, [], bias_edge, cast_bias_edge)])

        # Create the ADD node
        add_node = lgf_pb2.LNF()
        add_node.name = fully_connected_node.name + "_add"
        add_node.supported = True
        add_node.vv_add.SetInParent()
        # Create the ADD input data edge
        add_node.inputs.add().CopyFrom(matmul_node.outputs[0])
        # Create the ADD input bias edge
        add_bias_edge = add_node.inputs.add()
        add_bias_edge.CopyFrom(cast_bias_edge)
        add_bias_edge.dtype.CopyFrom(self._sw_config.float_type)
        # Create the ADD output edge
        add_output_edge = add_node.outputs.add()
        add_output_edge.name = add_node.name
        add_output_edge.port = 0
        add_output_edge.dtype.CopyFrom(add_node.inputs[0].dtype)
        add_output_edge.shape.CopyFrom(add_node.inputs[0].shape)
        # Append the node to add
        to_add.append(add_node)
        to_reroute.extend([(edge_reroute, [], output_edge, add_output_edge)])

        # Create the CAST node to cast the bias to the type of the matmul
        cast_add_node = lgf_pb2.LNF()
        cast_add_node.name = fully_connected_node.name + "_cast_add"
        cast_add_node.supported = True
        cast_add_node.cast.SetInParent()
        # Create the CAST input edge
        cast_add_node.inputs.add().CopyFrom(add_output_edge)
        # Create the CAST output edge
        cast_add_edge = cast_add_node.outputs.add()
        cast_add_edge.name = cast_add_node.name
        cast_add_edge.port = 0
        cast_add_edge.dtype.CopyFrom(input_edge.dtype)
        cast_add_edge.shape.CopyFrom(add_output_edge.shape)
        # Append the node to add
        to_add.append(cast_add_node)
        # Append the edges to be rerouted: ADD>OUTPUT to CAST_ADD>OUTPUT
        to_reroute.extend([(edge_reroute, [], add_output_edge, cast_add_edge)])

        # Return the transforms
        return self.create_transform_result(to_add=to_add,
                                            to_replace=to_replace,
                                            to_reroute=to_reroute)
