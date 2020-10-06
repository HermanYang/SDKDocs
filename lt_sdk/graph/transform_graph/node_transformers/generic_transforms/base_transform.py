import numpy as np

from lt_sdk.graph.transform_graph import utils
from lt_sdk.graph.transform_graph.node_transformers import node_transform
from lt_sdk.proto import dtypes_pb2, lgf_pb2, transform_result_pb2


class BaseTransform(node_transform.NodeTransform):
    """
    Interface to implement generic versions of node transforms
    """

    def create_supported_nodes(self, *args, **kwargs):
        """Returns a list of supported nodes necessary for the generic transform"""
        raise NotImplementedError()

    def do_generic_transform(self, *args, **kwargs):
        """
        Returns a TransformResult to perform a generic transformation
        (by generic we mean independent of any third party library format)
        """
        # Makes assumptions about order of nodes returned by supported_nodes
        # override to perform more complicated transformations
        supported_nodes = self.create_supported_nodes(*args, **kwargs)
        return self.create_transform_result(to_replace=supported_nodes[0:1],
                                            to_add=supported_nodes[1:])

    def can_transform(self, node, light_graph):
        return True

    def _get_qint_dtype(self, precision, signed):
        """Returns the qint/quint dtype for the precision"""
        dtype = dtypes_pb2.DType()
        dtype.p = precision
        if signed:
            dtype.t = dtypes_pb2.DT_QINT
        else:
            dtype.t = dtypes_pb2.DT_QUINT
        return dtype

    def _get_dtype(self):
        if self._is_quantized():
            dtype = self._get_qint_dtype(
                self._sw_config.quantized_electronic_op_precision,
                True)
        else:
            dtype = self._sw_config.float_type
        return dtype

    def _is_quantized(self):
        return False

    def _get_input_index(self, child_node, parent_node):
        """
        Returns the index of child_node in parent_node.inputs
        Assumes exactly one output port from child_node is an input to parent_node
        """
        for i, e in enumerate(parent_node.inputs):
            if e.name == child_node.name:
                return i

        raise ValueError("{0} is not found in the inputs of {1}".format(
            child_node.name,
            parent_node.name))

    @staticmethod
    def create_const_node(array, name, dtype, const_type):
        const_node = lgf_pb2.LNF()
        const_node.name = name
        const_node.supported = True
        const_node.const.SetInParent()
        const_node.const.value.CopyFrom(utils.array_to_tensor_pb(array, dtype))
        const_node.const.const_type = const_type

        output_edge = const_node.outputs.add()
        output_edge.name = const_node.name
        output_edge.port = 0
        output_edge.dtype.CopyFrom(const_node.const.value.dtype)
        output_edge.shape.CopyFrom(const_node.const.value.shape)

        return const_node

    @staticmethod
    def create_image_attr(kernel_size, strides, padding, data_format):
        # Fix defaults
        if isinstance(padding, str):
            padding = lgf_pb2.ImagePatchAttributes.Padding.Value(padding)
        if isinstance(data_format, str):
            data_format = lgf_pb2.ImagePatchAttributes.DataFormat.Value(data_format)

        if data_format == lgf_pb2.ImagePatchAttributes.NHWC:
            h_index = 1
            w_index = 2
        elif data_format == lgf_pb2.ImagePatchAttributes.NCHW:
            h_index = 2
            w_index = 3
        else:
            raise ValueError("Invalid data format {}".format(data_format))

        if len(kernel_size) == 2:
            h, w = kernel_size
            kernel_size = [1, 1, 1, 1]
            kernel_size[h_index] = h
            kernel_size[w_index] = w
        if len(strides) == 2:
            h, w = strides
            strides = [1, 1, 1, 1]
            strides[h_index] = h
            strides[w_index] = w

        assert (len(kernel_size) == 4)
        assert (len(strides) == 4)

        # Create proto
        image_attr = lgf_pb2.ImagePatchAttributes()
        image_attr.kernel_size.extend(kernel_size)
        image_attr.strides.extend(strides)
        image_attr.padding = padding
        image_attr.data_format = data_format

        return image_attr

    @staticmethod
    def check_original_node(node, graph_type=None):
        assert (not node.supported)
        assert (node.HasField(lgf_pb2.LNF.original.DESCRIPTOR.name))
        if graph_type is not None:
            assert (node.original.t == graph_type)

    @staticmethod
    def get_array_from_const_node(const_node):
        if const_node.HasField(lgf_pb2.LNF.const.DESCRIPTOR.name):
            const = const_node.const
        else:
            raise ValueError("Invalid const node")

        return utils.tensor_pb_to_array(const.value, np.float32)

    @staticmethod
    def find_input_and_weight_nodes(node, light_graph):
        """Returns a tuple of lgf_pb2.LNF's for (input, weights)"""
        inp_node = None
        weight_node = None

        # Try to set weight to constant node
        for edge in node.inputs:
            n = light_graph.get_node_by_name(edge.name)
            if light_graph.is_constant_node(n):
                weight_node = n
            else:
                inp_node = n

        # If we could not find weight node, just arbitrarily set
        # weights and inputs
        if weight_node is None:
            if len(node.inputs) < 2:
                raise ValueError(
                    "Could not find input and weight nodes for node {}".format(node))
            inp_node = light_graph.get_node_by_name(node.inputs[0].name)
            weight_node = light_graph.get_node_by_name(node.inputs[1].name)

        return inp_node, weight_node

    @staticmethod
    def create_transform_result(to_add=[],
                                to_replace=[],
                                to_reroute=[],
                                to_output_swap=[]):
        """
        Params:
            to_add: a list of lgf_pb2.LNF() nodes
            to_replace: a list of lgf_pb2.LNF() nodes
            to_reroute: a list of tuples, tuples are of the form
                (reroute type, dst_node_names, reroute param 0, ..., reroute param n - 1)
                dst_node_names is a list/tuple of node names
                reroute param i corresponds to field i + 1 in the reroute type message
            to_output_swap: a list of tuples, tuples are of the form
                (old_output_node_names, new_output_node_names)
                old_output_node_names and new_output_node_names are each a lists/tuple
                of node names

        Returns:
            a transform_result_pb2.TransformResult()
        """
        transform_result = transform_result_pb2.TransformResult()

        for node in to_add:
            transform_result.to_add.add().node.CopyFrom(node)

        for node in to_replace:
            transform_result.to_replace.add().node.CopyFrom(node)

        for tup in to_reroute:
            reroute_type = tup[0]
            dst_node_names = tup[1]
            reroute_transform = transform_result.to_reroute.add()
            reroute_transform.dst_node_names.extend(dst_node_names)

            reroute = getattr(reroute_transform, reroute_type)
            reroute.SetInParent()
            for i, field_name in enumerate(reroute.DESCRIPTOR.fields_by_name):
                field_res = tup[i + 2]
                try:
                    # Singular field
                    setattr(reroute, field_name, field_res)
                except AttributeError:
                    pass
                try:
                    # List field
                    getattr(reroute, field_name).extend(field_res)
                except AttributeError:
                    # Message field
                    getattr(reroute, field_name).CopyFrom(field_res)

        for tup in to_output_swap:
            output_swap_transform = transform_result.to_output_swap.add()
            output_swap_transform.old_output_node_names.extend(tup[0])
            output_swap_transform.new_output_node_names.extend(tup[1])

        return transform_result

    @staticmethod
    def _get_height_and_width_indices(image_attr):
        if image_attr.data_format == lgf_pb2.ImagePatchAttributes.NHWC:
            return 1, 2, 3
        elif image_attr.data_format == lgf_pb2.ImagePatchAttributes.NCHW:
            return 2, 3, 1
        else:
            raise ValueError("Invalid data format")
