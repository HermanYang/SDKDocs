from sdk2.graph.transform_graph.node_transformers.generic_transforms import (
    identity_transform, relu6_transform, reshape_transform, sigmoid_transform,
    sv_transform, swish_transform, tanh_transform, tile_transform, vv_transform)
from sdk2.graph.transform_graph.node_transformers.tf_transforms import (
    tf_batch_matmul_transform, tf_batch_norm_transform, tf_concat_transform,
    tf_const_transform, tf_conv2d_transform, tf_fill_transform, tf_matmul_transform,
    tf_mean_transform, tf_pad_transform, tf_pool_transform, tf_pow_transform,
    tf_softmax_transform, tf_split_transform, tf_squared_difference_transform,
    tf_squeeze_transform, tf_stack_transform, tf_transpose_transform,
    tf_unstack_transform)
from sdk2.graph.transform_graph.node_transformers.tflite_transforms import (
    tflite_const_transform, tflite_dequantize_transform,
    tflite_fully_connected_transform, tflite_quantize_transform)
from sdk2.proto import graph_types_pb2, ops_pb2

GENERIC_NODE_TRANSFORM_MAP = {
    ops_pb2.IDENTITY: identity_transform.IdentityTransform,
    ops_pb2.RESHAPE: reshape_transform.ReshapeTransform,
    ops_pb2.RELU: sv_transform.ReluTransform,
    ops_pb2.RELU6: relu6_transform.Relu6Transform,
    ops_pb2.ADD: vv_transform.VVAddTransform,
    ops_pb2.SWISH: swish_transform.SwishTransform,
    ops_pb2.MULTIPLY: vv_transform.VVMulTransform,
    ops_pb2.SIGMOID: sigmoid_transform.SigmoidTransform,
    ops_pb2.TANH: tanh_transform.TanhTransform,
    ops_pb2.SUB: vv_transform.VVSubTransform,
    ops_pb2.EXPANDDIMS: reshape_transform.ReshapeTransform,
    ops_pb2.RSQRT: sv_transform.RsqrtTransform,
    ops_pb2.TILE: tile_transform.TileTransform,
}

TF_NODE_TRANSFORM_MAP = {
    ops_pb2.AVGPOOL:
        tf_pool_transform.TFSavedModelPoolTransform,
    ops_pb2.BATCHNORM:
        tf_batch_norm_transform.TFSavedModelBatchNormTransform,
    ops_pb2.CONST:
        tf_const_transform.TFSavedModelConstTransform,
    ops_pb2.CONV2D:
        tf_conv2d_transform.TFSavedModelConv2DTransform,
    ops_pb2.DEPTHWISE_CONV2D:
        tf_conv2d_transform.TFSavedModelDepthwiseConv2DTransform,
    ops_pb2.MATMUL:
        tf_matmul_transform.TFSavedModelMatMulTransform,
    ops_pb2.MAXPOOL:
        tf_pool_transform.TFSavedModelPoolTransform,
    ops_pb2.MEAN:
        tf_mean_transform.TFSavedModelMeanTransform,
    ops_pb2.PAD:
        tf_pad_transform.TFSavedModelPadTransform,
    ops_pb2.SOFTMAX:
        tf_softmax_transform.TFSavedModelSoftmaxTransform,
    ops_pb2.SQUEEZE:
        tf_squeeze_transform.TFSavedModelSqueezeTransform,
    ops_pb2.TRANSPOSE:
        tf_transpose_transform.TFSavedModelTransposeTransform,
    ops_pb2.UNSTACK:
        tf_unstack_transform.TFSavedModelUnstackTransform,
    ops_pb2.POW:
        tf_pow_transform.TFSavedModelPowTransform,
    ops_pb2.FILL:
        tf_fill_transform.TFSavedModelFillTransform,
    ops_pb2.SQUARED_DIFFERENCE:
        tf_squared_difference_transform.TFSavedModelSquaredDifferenceTransform,
    ops_pb2.STACK:
        tf_stack_transform.TFSavedModelStackTransform,
    ops_pb2.BATCHMATMUL:
        tf_batch_matmul_transform.TFSavedModelBatchMatMulV2Transform,
    ops_pb2.CONCAT:
        tf_concat_transform.TFSavedModelConcatV2Transform,
    ops_pb2.SPLIT:
        tf_split_transform.TFSavedModelSplitTransform,
}

TFLITE_NODE_TRANSFORM_MAP = {
    ops_pb2.CONST:
        tflite_const_transform.TFLiteConstTransform,
    ops_pb2.DEQUANTIZE:
        tflite_dequantize_transform.TFLiteDequantizeTransform,
    ops_pb2.FULLY_CONNECTED:
        tflite_fully_connected_transform.TFLiteFullyConnectedTransform,
    ops_pb2.QUANTIZE:
        tflite_quantize_transform.TFLiteQuantizeTransform,
}

NODE_TRANSFORM_MAP = {
    graph_types_pb2.TFSavedModel: TF_NODE_TRANSFORM_MAP,
    graph_types_pb2.TFTrainingSavedModel: TF_NODE_TRANSFORM_MAP,
    graph_types_pb2.TFTrainingCheckpoint: TF_NODE_TRANSFORM_MAP,
    graph_types_pb2.TFGraphDef: TF_NODE_TRANSFORM_MAP,
    graph_types_pb2.TFLiteSavedModel: TFLITE_NODE_TRANSFORM_MAP,
}


def get_node_transform(graph_type, op):
    if graph_type not in NODE_TRANSFORM_MAP:
        raise ValueError("Unsupported graph type: {}".format(graph_type))

    specific_map = NODE_TRANSFORM_MAP[graph_type]
    if op in specific_map:
        return specific_map[op]
    elif op in GENERIC_NODE_TRANSFORM_MAP:
        return GENERIC_NODE_TRANSFORM_MAP[op]
    else:
        raise ValueError("Could not find transform for graph type {} and op {}".format(
            graph_type, op))
