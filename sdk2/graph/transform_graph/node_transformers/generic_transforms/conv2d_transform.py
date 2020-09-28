from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    opu_op_transform
from sdk2.proto import lgf_pb2


class Conv2DTransform(opu_op_transform.OPUOpTransform):

    def init_conv2d_node(self, conv2d_name):
        """
        Initializes a conv2d node

        Params:
            conv2d_name: name for the conv2d node

        Returns:
            a tuple: (a lgf_pb2.LNF() object for the node,
                a lgf_pb2.Conv2DNode() for the conv2d of the node)
        """
        # Create a conv2d_node
        conv2d_node = lgf_pb2.LNF()
        conv2d_node.name = conv2d_name
        conv2d_node.supported = True
        conv2d_node.conv2d.SetInParent()

        return conv2d_node, conv2d_node.conv2d

    def preprocess_weights(self, weights_edge):
        """
        Adds extra nodes to process the weights before sending to phasify

        Params:
            weight_edge: edge to be preprocessed

        Returns:
            a tuple: (preprocessing is possible,
                a list of nodes to do the preprocessing,
                the new output edge to feed into phasify)
        """
        return True, [], weights_edge

    def _try_to_convert_to_matmul(self, conv2d_node):
        if not (conv2d_node.HasField(lgf_pb2.LNF.conv2d.DESCRIPTOR.name)):
            return

        h_indx, w_indx, _ = self._get_height_and_width_indices(
            conv2d_node.conv2d.image_attr)
        input_shape = conv2d_node.inputs[lgf_pb2.MatMulNode.INPUT_INDEX].shape
        output_shape = conv2d_node.outputs[0].shape

        input_dim = input_shape.d[h_indx], input_shape.d[w_indx]
        output_dim = output_shape.d[h_indx], output_shape.d[w_indx]

        kernel_size = conv2d_node.conv2d.image_attr.kernel_size
        kernel_dim = kernel_size[h_indx], kernel_size[w_indx]
        strides = conv2d_node.conv2d.image_attr.strides
        stride_dim = strides[h_indx], strides[w_indx]

        # should only replace matmul if input and output h & w are the same,
        # kernel h & w are 1, and stride h & w are 1.
        if input_dim == output_dim and kernel_dim == (1, 1) and stride_dim == (1, 1):
            conv2d_node.matmul.CopyFrom(conv2d_node.conv2d.matmul)

    def create_supported_nodes(self, conv2d_name, input_edge, weights_edge, output_edge,
                               control_inputs, kernel_size, strides, padding,
                               data_format):
        """
        Creates a supported conv2d node in standard format

        Params:
            conv2d_name: name of original node
            input_edge: edge of the input for the original node
            weights_edge: input edge for the weights, to be fed to phasify
            output_edge: edge of the output for the original node
            control_inputs: a list of edges for the control inputs
            kernel_size: list or numpy array of 2 or 4 numbers
            strides: list or numpy array of 2 or 4 numbers
            padding: string or enum for padding
            data_format: string or enum for data format
        """
        # Create a conv2d_node
        conv2d_node, conv2d = self.init_conv2d_node(conv2d_name)

        # Preprocess the weights
        can_preprocess, preprocess_nodes, phase_input_edge = self.preprocess_weights(
            weights_edge)

        if not can_preprocess:
            return []

        # Phasify
        to_add = preprocess_nodes
        to_add += self._create_phasify_node(conv2d_node, phase_input_edge)

        # Attributes for conv2d
        conv2d.image_attr.CopyFrom(
            self.create_image_attr(kernel_size, strides, padding, data_format))

        # Input data
        conv2d_node.inputs[lgf_pb2.MatMulNode.INPUT_INDEX].CopyFrom(input_edge)
        conv2d_node.inputs[lgf_pb2.MatMulNode.INPUT_INDEX].dtype.CopyFrom(
            self._sw_config.float_type)
        conv2d_node.control_inputs.extend(control_inputs)

        # Outputs
        conv2d_node.outputs.add().CopyFrom(output_edge)
        conv2d_node.outputs[0].dtype.CopyFrom(self._sw_config.float_type)

        # Special case where conv2d can be a matmul
        self._try_to_convert_to_matmul(conv2d_node)

        return [conv2d_node] + to_add
