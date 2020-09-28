import argparse
import logging
import os

from sdk2.proto import (common_pb2, dtypes_pb2, graph_types_pb2, hardware_configs_pb2,
                        lgf_pb2, node_filters, ops_pb2, sw_config_pb2)
from sdk2.proto.configs import utils
from yis_sdk import instruction_pb2

GRAPH_TYPE_OPS = {
    graph_types_pb2.TFSavedModel: {
        ops_pb2.AVGPOOL,
        ops_pb2.CONV2D,
        ops_pb2.MATMUL,
        ops_pb2.DEPTHWISE_CONV2D,
        ops_pb2.ADD,
        ops_pb2.RELU,
        ops_pb2.RELU6,
        ops_pb2.RESHAPE,
        ops_pb2.MAXPOOL,
        ops_pb2.BATCHNORM,
        ops_pb2.PAD,
        ops_pb2.MEAN,
        ops_pb2.SOFTMAX,
        ops_pb2.SQUEEZE,
        ops_pb2.IDENTITY,
        ops_pb2.SWISH,
        ops_pb2.SIGMOID,
        ops_pb2.MULTIPLY,
        ops_pb2.TRANSPOSE,
        ops_pb2.TANH,
        ops_pb2.UNSTACK,
        ops_pb2.SUB,
        ops_pb2.POW,
        ops_pb2.EXPANDDIMS,
        ops_pb2.FILL,
        ops_pb2.RSQRT,
        ops_pb2.SQUARED_DIFFERENCE,
        ops_pb2.STACK,
        ops_pb2.BATCHMATMUL,
        ops_pb2.TILE,
        ops_pb2.CONCAT,
        ops_pb2.SPLIT,
    },
    graph_types_pb2.TFLiteSavedModel: {
        ops_pb2.ADD,
        ops_pb2.CONST,
        ops_pb2.DEQUANTIZE,
        ops_pb2.FULLY_CONNECTED,
        ops_pb2.QUANTIZE,
        ops_pb2.IDENTITY,
    },
    graph_types_pb2.TFTrainingCheckpoint: {
        ops_pb2.CONV2D,
        ops_pb2.MATMUL,
        ops_pb2.DEPTHWISE_CONV2D,
    },
    graph_types_pb2.TFTrainingSavedModel: {
        ops_pb2.CONV2D,
        ops_pb2.MATMUL,
        ops_pb2.DEPTHWISE_CONV2D,
    },
    graph_types_pb2.ONNXModel: set()
}

GRAPH_TYPE_OPS[graph_types_pb2.TFGraphDef] = GRAPH_TYPE_OPS[graph_types_pb2.TFSavedModel]

HARDWARE_CONFIG_OPS = {
    hardware_configs_pb2.DELTA: {
        ops_pb2.AVGPOOL,
        ops_pb2.CONV2D,
        ops_pb2.MATMUL,
        ops_pb2.DEPTHWISE_CONV2D,
        ops_pb2.ADD,
        ops_pb2.RELU,
        ops_pb2.RELU6,
        ops_pb2.RESHAPE,
        ops_pb2.MAXPOOL,
        ops_pb2.BATCHNORM,
        ops_pb2.PAD,
        ops_pb2.MEAN,
        ops_pb2.SOFTMAX,
        ops_pb2.SQUEEZE,
        ops_pb2.IDENTITY,
        ops_pb2.SWISH,
        ops_pb2.SIGMOID,
        ops_pb2.MULTIPLY,
        ops_pb2.TANH,
        ops_pb2.SUB,
        ops_pb2.EXPANDDIMS,
        ops_pb2.FILL,
    },
    hardware_configs_pb2.VANGUARD: {
        ops_pb2.AVGPOOL,
        ops_pb2.CONV2D,
        ops_pb2.MATMUL,
        ops_pb2.DEPTHWISE_CONV2D,
        ops_pb2.ADD,
        ops_pb2.RELU,
        ops_pb2.RELU6,
        ops_pb2.RESHAPE,
        ops_pb2.MAXPOOL,
        ops_pb2.BATCHNORM,
        ops_pb2.PAD,
        ops_pb2.MEAN,
        ops_pb2.SOFTMAX,
        ops_pb2.SQUEEZE,
        ops_pb2.IDENTITY,
        ops_pb2.SWISH,
        ops_pb2.SIGMOID,
        ops_pb2.MULTIPLY,
        ops_pb2.TRANSPOSE,
        ops_pb2.TANH,
        ops_pb2.UNSTACK,
        ops_pb2.SUB,
        ops_pb2.POW,
        ops_pb2.EXPANDDIMS,
        ops_pb2.FILL,
        ops_pb2.RSQRT,
        ops_pb2.SQUARED_DIFFERENCE,
        ops_pb2.STACK,
        ops_pb2.BATCHMATMUL,
        ops_pb2.TILE,
        ops_pb2.CONCAT,
        ops_pb2.SPLIT,
    }
}

HARDWARE_CONFIG_OPS[hardware_configs_pb2.BRAVO] = HARDWARE_CONFIG_OPS[
    hardware_configs_pb2.DELTA]


def add_op_transform(sw_config, graph_type, op, ignore_nodes_filter):
    pair = sw_config.filter_transform_map.add()

    filt = node_filters.op_filter(op)
    filt = node_filters.and_filter(filt, node_filters.not_filter(ignore_nodes_filter))

    pair.filter.CopyFrom(filt.as_proto())
    pair.transform.graph_type = graph_type
    pair.transform.op = op


def get_standard_filter_transform_map(sw_config, graph_type, hw_cfg,
                                      ignore_nodes_filter):
    # Don't add transformations to a fully supported graph
    if graph_type == graph_types_pb2.LGFProtobuf:
        return

    # Otherwise add standard op transforms
    for op in GRAPH_TYPE_OPS[graph_type].intersection(HARDWARE_CONFIG_OPS[hw_cfg]):
        add_op_transform(sw_config, graph_type, op, ignore_nodes_filter)


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
    import sdk2
    module_path = os.path.dirname(sdk2.__file__)
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
        float_type=create_dtype(dtypes_pb2.DT_BFLOAT, 16),
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
        lgf_pb2.LNF.matmul.DESCRIPTOR.name, lgf_pb2.LNF.conv2d.DESCRIPTOR.name,
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
        generate_standard_sw_config(graph_types_pb2.TFSavedModel, protos=protos),
        os.path.join(out_dir, "standard.sw_config.pb.txt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",
                        default=os.path.join(os.path.expanduser("~"), "sw_configs"),
                        type=str,
                        help="Name of set of sw configs.")
    parser.add_argument("--protos", nargs="*", type=str, help="Protos we might need")
    args = parser.parse_args()
    generate(out_dir=args.out_dir, protos=args.protos)
