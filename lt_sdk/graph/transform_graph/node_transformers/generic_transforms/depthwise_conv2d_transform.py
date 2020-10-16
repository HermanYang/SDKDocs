import logging

import numpy as np

from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    conv2d_transform,
    reshape_transform,
    transpose_transform,
)
from lt_sdk.proto import hw_spec_pb2, lgf_pb2


class BlockDiagionalDepthwiseConv2DTransform(conv2d_transform.Conv2DTransform):

    ARCH_TYPES = {
        hw_spec_pb2.HardwareSpecs.coherent.DESCRIPTOR.name,
        hw_spec_pb2.HardwareSpecs.ideal_incoherent.DESCRIPTOR.name,
        hw_spec_pb2.HardwareSpecs.physical_incoherent.DESCRIPTOR.name,
    }

    def init_conv2d_node(self, conv2d_name):
        # Create a block diagonal depthwise conv2d node
        depthwise_conv2d_node = lgf_pb2.LNF()
        depthwise_conv2d_node.name = conv2d_name
        depthwise_conv2d_node.supported = True
        depthwise_conv2d_node.block_diagonal_depthwise_conv2d.SetInParent()

        return (depthwise_conv2d_node,
                depthwise_conv2d_node.block_diagonal_depthwise_conv2d.conv2d)

    def preprocess_weights(self, weights_edge):
        # Unpack shapes
        filter_height, filter_width, in_channels, channel_multiplier = \
            weights_edge.shape.d
        k = self._hw_specs.dimension
        j = self.num_columns()

        # There are in_channels matrices, each of size
        # [filter_height * filter_width, channel_multiplier]
        # Figure out how many we can stack on each axis
        x_stack = k // (filter_height * filter_width)
        y_stack = j // (channel_multiplier)
        matrices_per_tile = min(x_stack, y_stack)

        # If the matrices are too big to fit into the matmul unit, we cannot
        # do the transform
        if not matrices_per_tile:
            logging.error("Failed to transform Block Diagonal Depthwise Conv2d")
            return False, [], weights_edge

        # Number of opu tiles we will need for in_channels matrices
        num_tiles = np.ceil(in_channels / matrices_per_tile).astype(int)

        # Create depthwise conv2d reshape node
        depthwise_conv2d_reshape_node = lgf_pb2.LNF()
        depthwise_conv2d_reshape_node.name = weights_edge.name + "_depthwise_reshape"
        depthwise_conv2d_reshape_node.supported = True
        (depthwise_conv2d_reshape_node.block_diagonal_depthwise_conv2d_reshape.
         SetInParent())

        input_edge = depthwise_conv2d_reshape_node.inputs.add()
        input_edge.CopyFrom(weights_edge)
        input_edge.dtype.CopyFrom(self._get_dtype())

        output_edge = depthwise_conv2d_reshape_node.outputs.add()
        output_edge.name = depthwise_conv2d_reshape_node.name
        output_edge.port = 0
        output_edge.dtype.CopyFrom(input_edge.dtype)
        output_edge.shape.d.extend([k, num_tiles * j])
        output_edge.shape.batch_dim_indx = -1

        return True, [depthwise_conv2d_reshape_node], output_edge


class DistributedDepthwiseConv2D(conv2d_transform.Conv2DTransform,
                                 transpose_transform.TransposeTransform,
                                 reshape_transform.ReshapeTransform):

    ARCH_TYPES = {
        hw_spec_pb2.HardwareSpecs.ideal_digital.DESCRIPTOR.name,
        hw_spec_pb2.HardwareSpecs.physical_digital.DESCRIPTOR.name,
    }

    def init_conv2d_node(self, conv2d_name):
        # Create a distributed depthwise conv2d node
        depthwise_conv2d_node = lgf_pb2.LNF()
        depthwise_conv2d_node.name = conv2d_name
        depthwise_conv2d_node.supported = True
        depthwise_conv2d_node.distributed_depthwise_conv2d.SetInParent()

        return (depthwise_conv2d_node,
                depthwise_conv2d_node.distributed_depthwise_conv2d.conv2d)

    def preprocess_weights(self, weights_edge):
        # Unpack shapes
        filter_height, filter_width, in_channels, channel_multiplier = \
            weights_edge.shape.d

        # Transpose to [filter_height, filter_width, channel_multiplier, in_channels]
        transpose_edge = lgf_pb2.EdgeInfo()
        transpose_edge.CopyFrom(weights_edge)
        transpose_edge.name = weights_edge.name + "_transpose"
        transpose_edge.shape.d[:] = [
            filter_height,
            filter_width,
            channel_multiplier,
            in_channels
        ]
        transpose_node = transpose_transform.TransposeTransform.create_supported_nodes(
            self,
            transpose_edge.name,
            weights_edge,
            transpose_edge,
            [],
            [0,
             1,
             3,
             2])[0]

        # Reshape to [filter_height * filter_width, channel_multiplier * in_channels]
        reshape_edge = lgf_pb2.EdgeInfo()
        reshape_edge.CopyFrom(transpose_edge)
        reshape_edge.name = transpose_edge.name + "_reshape"
        reshape_edge.shape.d[:] = [
            filter_height * filter_width,
            channel_multiplier * in_channels
        ]
        reshape_node = reshape_transform.ReshapeTransform.create_supported_nodes(
            self,
            reshape_edge.name,
            transpose_node.outputs[0],
            reshape_edge,
            [])[0]

        return True, [transpose_node, reshape_node], reshape_node.outputs[0]


class DepthwiseConv2DTransform(BlockDiagionalDepthwiseConv2DTransform,
                               DistributedDepthwiseConv2D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        arch = self._hw_specs.WhichOneof("architecture")
        if arch in BlockDiagionalDepthwiseConv2DTransform.ARCH_TYPES:
            self._cls = BlockDiagionalDepthwiseConv2DTransform
        elif arch in DistributedDepthwiseConv2D.ARCH_TYPES:
            self._cls = DistributedDepthwiseConv2D
        else:
            raise ValueError("Could not find Depthwise Conv2D transform for the given" +
                             " architecture: {}".format(arch))

    def init_conv2d_node(self, conv2d_name):
        return self._cls.init_conv2d_node(self, conv2d_name)

    def preprocess_weights(self, weights_edge):
        return self._cls.preprocess_weights(self, weights_edge)
