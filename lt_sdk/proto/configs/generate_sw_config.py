import argparse
import copy
import logging
import os

from lt_sdk.proto import (
    common_pb2,
    dtypes_pb2,
    graph_types_pb2,
    hardware_configs_pb2,
    lgf_pb2,
    node_filters,
    ops_pb2,
    sw_config_pb2,
)
from lt_sdk.proto.configs import utils
from yis_sdk import instruction_pb2

# This map indicates which transforms to use for both graph types and the target
# hardware configuration
OP_TRANSFORMERS = {
    hardware_configs_pb2.DELTA: {},
    hardware_configs_pb2.VANGUARD: {},
    hardware_configs_pb2.BRAVO: {}
}

# Const transform is special, we selectively transform them in
# mark_supported_const
CONST_TRANSFORM = "tf_const_transform.TFSavedModelConstTransform"

# ----- Generic transformers ------ #
# Generic transforms are not framework-specific.
# These usually don't extract any parameters, such as NN activations.
BASE_GENERIC_TRANSFORMS = {
    # Keep these alphabetical
    ops_pb2.ADD: "vv_transform.VVAddTransform",
    ops_pb2.EXPANDDIMS: "reshape_transform.ReshapeTransform",
    ops_pb2.IDENTITY: "identity_transform.IdentityTransform",
    ops_pb2.MULTIPLY: "vv_transform.VVMulTransform",
    ops_pb2.RELU: "sv_transform.ReluTransform",
    ops_pb2.RELU6: "relu6_transform.Relu6Transform",
    ops_pb2.RESHAPE: "reshape_transform.ReshapeTransform",
    ops_pb2.SIGMOID: "sigmoid_transform.SigmoidTransform",
    ops_pb2.SWISH: "swish_transform.SwishTransform",
    ops_pb2.TANH: "tanh_transform.TanhTransform",
}

EXTENDED_GENERIC_TRANSFORMS = copy.copy(BASE_GENERIC_TRANSFORMS)
EXTENDED_GENERIC_TRANSFORMS.update({
    ops_pb2.RSQRT: "sv_transform.RsqrtTransform",
    ops_pb2.SUB: "vv_transform.VVSubTransform",
    ops_pb2.TILE: "tile_transform.TileTransform",
})

# ----- TF Saved Model transformers ------ #
OP_TRANSFORMERS[hardware_configs_pb2.DELTA][graph_types_pb2.TFSavedModel] = copy.copy(
    BASE_GENERIC_TRANSFORMS)
OP_TRANSFORMERS[hardware_configs_pb2.DELTA][graph_types_pb2.TFSavedModel].update({
    # Keep these alphabetical
    ops_pb2.BATCHNORM:
        "tf_batch_norm_transform.TFSavedModelDecomposeBatchNormTransform",
    ops_pb2.CONV2D:
        "tf_conv2d_transform.TFSavedModelConv2DTransform",
    ops_pb2.FILL:
        "tf_fill_transform.TFSavedModelFillTransform",
    ops_pb2.MATMUL:
        "tf_matmul_transform.TFSavedModelMatMulTransform",
    ops_pb2.DEPTHWISE_CONV2D:
        "tf_conv2d_transform.TFSavedModelDepthwiseConv2DTransform",
    ops_pb2.MAXPOOL:
        "tf_pool_transform.TFSavedModelPoolTransform",
    ops_pb2.MEAN:
        "tf_mean_transform.TFSavedModelMeanTransform",
    ops_pb2.PAD:
        "tf_pad_transform.TFSavedModelUnsupportedPadTransform",
    ops_pb2.SOFTMAX:
        "tf_softmax_transform.TFSavedModelSoftmaxTransform",
    ops_pb2.SQUEEZE:
        "tf_squeeze_transform.TFSavedModelSqueezeTransform",
})

# Vanguard is a superset of Delta
OP_TRANSFORMERS[hardware_configs_pb2.VANGUARD][graph_types_pb2.TFSavedModel] = copy.copy(
    OP_TRANSFORMERS[hardware_configs_pb2.DELTA][graph_types_pb2.TFSavedModel])
OP_TRANSFORMERS[hardware_configs_pb2.VANGUARD][graph_types_pb2.TFSavedModel].update(
    EXTENDED_GENERIC_TRANSFORMS)

# More transforms supported by Vanguard
OP_TRANSFORMERS[hardware_configs_pb2.VANGUARD][graph_types_pb2.TFSavedModel].update({
    # Keep these alphabetical
    ops_pb2.AVGPOOL:
        "tf_pool_transform.TFSavedModelPoolTransform",
    ops_pb2.BATCHMATMUL:
        "tf_batch_matmul_transform.TFSavedModelBatchMatMulV2Transform",
    ops_pb2.CONCAT:
        "tf_concat_transform.TFSavedModelConcatV2Transform",
    ops_pb2.PAD:
        "tf_pad_transform.TFSavedModelPadTransform",
    ops_pb2.POW:
        "tf_pow_transform.TFSavedModelPowTransform",
    ops_pb2.SQUARED_DIFFERENCE:
        "tf_squared_difference_transform.TFSavedModelSquaredDifferenceTransform",
    ops_pb2.SPLIT:
        "tf_split_transform.TFSavedModelSplitTransform",
    ops_pb2.STACK:
        "tf_stack_transform.TFSavedModelStackTransform",
    ops_pb2.TRANSPOSE:
        "tf_transpose_transform.TFSavedModelTransposeTransform",
    ops_pb2.UNSTACK:
        "tf_unstack_transform.TFSavedModelUnstackTransform",
})

# ----- TF GraphDef transformers ------ #
OP_TRANSFORMERS[hardware_configs_pb2.DELTA][graph_types_pb2.TFGraphDef] = copy.deepcopy(
    OP_TRANSFORMERS[hardware_configs_pb2.DELTA][graph_types_pb2.TFSavedModel])
OP_TRANSFORMERS[hardware_configs_pb2.VANGUARD][
    graph_types_pb2.TFGraphDef] = copy.deepcopy(
        OP_TRANSFORMERS[hardware_configs_pb2.VANGUARD][graph_types_pb2.TFSavedModel])

# ----- TF Training transformers ------ #
OP_TRANSFORMERS[hardware_configs_pb2.DELTA][graph_types_pb2.TFTrainingSavedModel] = {
    ops_pb2.CONV2D:
        OP_TRANSFORMERS[hardware_configs_pb2.DELTA][graph_types_pb2.TFSavedModel]
        [ops_pb2.CONV2D],
    ops_pb2.MATMUL:
        OP_TRANSFORMERS[hardware_configs_pb2.DELTA][graph_types_pb2.TFSavedModel]
        [ops_pb2.MATMUL],
    ops_pb2.DEPTHWISE_CONV2D:
        OP_TRANSFORMERS[hardware_configs_pb2.DELTA][graph_types_pb2.TFSavedModel]
        [ops_pb2.DEPTHWISE_CONV2D],
}

OP_TRANSFORMERS[hardware_configs_pb2.DELTA][
    graph_types_pb2.TFTrainingCheckpoint] = copy.copy(OP_TRANSFORMERS[
        hardware_configs_pb2.DELTA][graph_types_pb2.TFTrainingSavedModel])

OP_TRANSFORMERS[hardware_configs_pb2.VANGUARD][
    graph_types_pb2.TFTrainingSavedModel] = copy.copy(OP_TRANSFORMERS[
        hardware_configs_pb2.DELTA][graph_types_pb2.TFTrainingSavedModel])

OP_TRANSFORMERS[hardware_configs_pb2.VANGUARD][
    graph_types_pb2.TFTrainingCheckpoint] = copy.copy(OP_TRANSFORMERS[
        hardware_configs_pb2.DELTA][graph_types_pb2.TFTrainingCheckpoint])

# ----- ONNX transformers ------ #
OP_TRANSFORMERS[hardware_configs_pb2.DELTA][graph_types_pb2.ONNXModel] = {}
OP_TRANSFORMERS[hardware_configs_pb2.VANGUARD][graph_types_pb2.ONNXModel] = {}

# ----- TF-Lite transformers ------ #
# TODO: re-enable these

# -----------------------------------------------#
OP_TRANSFORMERS[hardware_configs_pb2.BRAVO] = copy.deepcopy(
    OP_TRANSFORMERS[hardware_configs_pb2.DELTA])


def add_op_transform(sw_config, graph_type, op, ignore_nodes_filter, tx_name):
    pair = sw_config.filter_transform_map.add()

    filt = node_filters.op_filter(op)
    filt = node_filters.and_filter(filt, node_filters.not_filter(ignore_nodes_filter))

    pair.filter.CopyFrom(filt.as_proto())
    pair.transform.graph_type = graph_type
    pair.transform.op = op
    pair.transform.transform_module_name = tx_name


def fill_transform_map_with_ops(ops, sw_config, graph_type, hw_cfg, ignore_nodes_filter):
    for op in ops:
        if op in OP_TRANSFORMERS[hw_cfg][graph_type]:
            tx_name = OP_TRANSFORMERS[hw_cfg][graph_type][op]
        else:
            continue

        add_op_transform(sw_config, graph_type, op, ignore_nodes_filter, tx_name)


def get_all_ops(hw_cfg):
    all_ops = set()
    for op_dict in OP_TRANSFORMERS[hw_cfg].values():
        all_ops = all_ops.union(set(op_dict.keys()))
    return all_ops


def get_standard_filter_transform_map(sw_config,
                                      graph_type,
                                      hw_cfg,
                                      ignore_nodes_filter):
    # Don't add transformations to a fully supported graph
    if graph_type == graph_types_pb2.LGFProtobuf:
        return

    # Otherwise add standard op transforms. Get all ops first
    # so we can select specific implementations over generic (Any)

    fill_transform_map_with_ops(get_all_ops(hw_cfg),
                                sw_config,
                                graph_type,
                                hw_cfg,
                                ignore_nodes_filter)


def create_dtype(t, p):
    res = dtypes_pb2.DType()
    res.t = t
    res.p = p
    return res


def _read_proto_from_file(filename, proto):
    logging.info(f"Reading proto from file: {filename}")
    try:
        with open(filename, "rb") as f:
            text = f.read()
        proto.ParseFromString(text)
    except FileNotFoundError as e:
        logging.warning(f"Couldn't find {filename} {os.getcwd()}")
        raise e


def _set_default_compiler_params(sw_config,
                                 allow_tmem_fall_back=False,
                                 tile_inputs_for_accumulators=True,
                                 dep_pc_distance_precision=16,
                                 num_opu_tiles_precision=16,
                                 num_batch_tiles_precision=16,
                                 no_odd_image_dims_conv2d=False):
    compiler_params = sw_config.compiler_params

    compiler_params.allow_tmem_fall_back = allow_tmem_fall_back
    compiler_params.tile_inputs_for_accumulators = tile_inputs_for_accumulators

    binary_instruction_params = compiler_params.binary_instruction_params
    binary_instruction_params.dep_pc_distance_precision = dep_pc_distance_precision
    binary_instruction_params.num_opu_tiles_precision = num_opu_tiles_precision
    binary_instruction_params.num_batch_tiles_precision = num_batch_tiles_precision

    compiler_restrictions = compiler_params.compiler_restrictions
    compiler_restrictions.no_odd_image_dims_conv2d = no_odd_image_dims_conv2d


def set_instruction_formats(sw_config, protos=[]):
    import lt_sdk
    module_path = os.path.dirname(lt_sdk.__file__)
    filename = os.path.join(module_path, "proto/lisa_inst_yis_instruction.pb2")
    aif = instruction_pb2.AllInstructionFormat()
    try:
        _read_proto_from_file(filename, aif)
    except FileNotFoundError as e:
        if len(protos) > 0:
            _read_proto_from_file(protos[0], aif)
        else:
            raise e

    for i_fmt in aif.instruction_formats:
        cp_i_fmt = sw_config.instruction_formats.add()
        cp_i_fmt.CopyFrom(i_fmt)

    for instr_name, op_code in aif.instr_op_code_map.items():
        sw_config.op_code_map[instr_name] = op_code
        sw_config.rev_op_code_map[op_code] = instr_name

    for alu_op_name, op_code in aif.alu_op_code_map.items():
        sw_config.alu_op_code_map[alu_op_name] = op_code


def generate_standard_sw_config(
        graph_type,
        num_threads_scales=32,
        activation_scale_quantization_bias_type=common_pb2.QB_NONE,
        weight_quantization_type=common_pb2.QT_PER_COL_PER_TILE,
        weight_quantization_cutoff=0,
        adc_scale_quantization_type=common_pb2.QT_PER_COL_PER_TILE,
        activation_scale_quantization_method=common_pb2.QM_MIN_KL_DIVERGENCE,
        adc_scale_quantization_method=common_pb2.QM_MIN_TOTAL_VARIATION_DISTANCE,
        cache_dir="",
        skip_adc_cal=False,
        skip_activation_cal=False,
        activation_scale_num_bins=4096,
        adc_scale_num_bins=4096,
        nodes_to_skip=[],
        float_type=create_dtype(dtypes_pb2.DT_BFLOAT,
                                16),
        convert_graph_to_debug_mode=False,
        debug_dir="",
        save_hist_html_files=False,
        ops_to_skip=[],
        fold_phasify=True,
        collect_bit_activity=False,
        collect_memory_layout=False,
        ignore_nodes_filter=None,
        ignore_empty_histograms=False,
        num_fine_tuning_epochs=0,
        py_batch_size=0,
        num_py_batches=0,
        use_unsigned_quant_scheme=False,
        quantized_electronic_nodes=[],
        allow_tmem_fall_back=False,
        tile_inputs_for_accumulators=True,
        dep_pc_distance_precision=16,
        num_opu_tiles_precision=16,
        num_batch_tiles_precision=16,
        protos=[],
        no_odd_image_dims_conv2d=False,
        disable_block_sparsity=True,
        hw_cfg=hardware_configs_pb2.VANGUARD):
    """
    Note that nodes_to_skip and ops_to_skip will be added to the
    ignore_node_node filter
    """
    # Default ignore_nodes_filter
    if ignore_nodes_filter is None:
        ignore_nodes_filter = node_filters.not_filter(node_filters.true_filter())

    # Update ignore_nodes_filter
    nodes_to_skip_filter = node_filters.or_filter(
        *[node_filters.name_is_filter(node_name) for node_name in nodes_to_skip])
    ops_to_skip_filter = node_filters.or_filter(
        *[node_filters.op_filter(op) for op in ops_to_skip])
    ignore_nodes_filter = node_filters.or_filter(ignore_nodes_filter,
                                                 nodes_to_skip_filter,
                                                 ops_to_skip_filter)

    # Initialize sw config
    sw_config = sw_config_pb2.SoftwareConfig()

    # Transform stages
    sw_config.standard_transform_stages.append(sw_config_pb2.BASE_TRANSFORMS)
    if not skip_activation_cal and graph_type != graph_types_pb2.TFLiteSavedModel:
        sw_config.standard_transform_stages.append(
            sw_config_pb2.ACTIVATION_SCALE_CALIBRATION)
    if not skip_adc_cal:
        sw_config.standard_transform_stages.append(sw_config_pb2.ADC_SCALE_CALIBRATION)
    if fold_phasify:
        sw_config.standard_transform_stages.append(sw_config_pb2.FOLD_PHASIFY_CONSTANTS)

    sw_config.use_weight_sharing = False

    sw_config.float_type.CopyFrom(float_type)
    sw_config.quantized_electronic_op_precision = 8

    sw_config.activation_scale_quantization_bias_type = \
        activation_scale_quantization_bias_type
    sw_config.weight_quantization_type = weight_quantization_type
    sw_config.weight_quantization_cutoff = weight_quantization_cutoff
    sw_config.adc_scale_quantization_type = adc_scale_quantization_type

    sw_config.activation_scale_quantization_method = activation_scale_quantization_method
    sw_config.adc_scale_quantization_method = adc_scale_quantization_method
    sw_config.ignore_empty_histograms = ignore_empty_histograms

    sw_config.activation_scale_num_bins = activation_scale_num_bins
    sw_config.adc_scale_num_bins = adc_scale_num_bins

    get_standard_filter_transform_map(sw_config, graph_type, hw_cfg, ignore_nodes_filter)

    sw_config.node_types.opu_nodes.extend([
        lgf_pb2.LNF.matmul.DESCRIPTOR.name,
        lgf_pb2.LNF.conv2d.DESCRIPTOR.name,
        lgf_pb2.LNF.block_diagonal_depthwise_conv2d.DESCRIPTOR.name,
        lgf_pb2.LNF.distributed_depthwise_conv2d.DESCRIPTOR.name
    ])
    sw_config.node_types.quantized_electronic_nodes.extend(quantized_electronic_nodes)

    sw_config.num_threads_scales = num_threads_scales

    sw_config.debug_info.collect_checksums = False
    sw_config.debug_info.debug_dir = debug_dir

    sw_config.max_proto_size = int(1.8e9)

    sw_config.cache_dir = cache_dir

    sw_config.sweep_info.py_batch_size = py_batch_size
    sw_config.sweep_info.num_py_batches = num_py_batches
    sw_config.sweep_info.convert_graph_to_debug_mode = convert_graph_to_debug_mode
    sw_config.sweep_info.save_hist_html_files = save_hist_html_files
    sw_config.sweep_info.collect_bit_activity = collect_bit_activity
    sw_config.sweep_info.collect_memory_layout = collect_memory_layout
    sw_config.sweep_info.num_fine_tuning_epochs = num_fine_tuning_epochs

    sw_config.ignore_nodes_filter.CopyFrom(ignore_nodes_filter.as_proto())

    sw_config.use_unsigned_quant_scheme = use_unsigned_quant_scheme

    _set_default_compiler_params(
        sw_config,
        allow_tmem_fall_back=allow_tmem_fall_back,
        tile_inputs_for_accumulators=tile_inputs_for_accumulators,
        dep_pc_distance_precision=dep_pc_distance_precision,
        num_opu_tiles_precision=num_opu_tiles_precision,
        num_batch_tiles_precision=num_batch_tiles_precision,
        no_odd_image_dims_conv2d=no_odd_image_dims_conv2d)

    set_instruction_formats(sw_config, protos)

    sw_config.disable_block_sparsity = disable_block_sparsity

    sw_config.const_transform_name = CONST_TRANSFORM

    return sw_config


def generate_mosaic_delta(graph_type,
                          skip_adc_cal=True,
                          quantized_electronic_nodes=[lgf_pb2.LNF.pool.DESCRIPTOR.name],
                          num_batch_tiles_precision=4,
                          dep_pc_distance_precision=16,
                          num_opu_tiles_precision=7,
                          no_odd_image_dims_conv2d=True,
                          disable_block_sparsity=False,
                          **kwargs):
    return generate_standard_sw_config(
        graph_type,
        skip_adc_cal=skip_adc_cal,
        quantized_electronic_nodes=quantized_electronic_nodes,
        num_batch_tiles_precision=num_batch_tiles_precision,
        dep_pc_distance_precision=dep_pc_distance_precision,
        num_opu_tiles_precision=num_opu_tiles_precision,
        no_odd_image_dims_conv2d=no_odd_image_dims_conv2d,
        disable_block_sparsity=disable_block_sparsity,
        hw_cfg=hardware_configs_pb2.DELTA,
        **kwargs)


def generate_vanguard(graph_type, skip_adc_cal=True, **kwargs):
    return generate_standard_sw_config(
        graph_type,
        skip_adc_cal=skip_adc_cal,
        hw_cfg=hardware_configs_pb2.VANGUARD,
        **kwargs,
    )


def generate(out_dir=None, protos=[]):
    out_dir = out_dir or os.path.dirname(__file__)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    utils.write_sw_config(
        generate_standard_sw_config(graph_types_pb2.TFSavedModel,
                                    protos=protos),
        os.path.join(out_dir,
                     "standard.sw_config.pb.txt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",
                        default=os.path.join(os.path.expanduser("~"),
                                             "sw_configs"),
                        type=str,
                        help="Name of set of sw configs.")
    parser.add_argument("--protos", nargs="*", type=str, help="Protos we might need")
    args = parser.parse_args()
    generate(out_dir=args.out_dir, protos=args.protos)
