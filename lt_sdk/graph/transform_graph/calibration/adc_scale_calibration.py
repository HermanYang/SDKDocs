import logging

from lt_sdk.graph.graph_collections import graph_collection
from lt_sdk.graph.run_graph import histogram_graph_runner
from lt_sdk.graph.transform_graph.calibration import (
    convert_to_adc_scale_calibration_graph,
)
from lt_sdk.graph.transform_graph.graph_transformers import apply_node_map
from lt_sdk.proto import calibration_pb2, common_pb2, sim_params_pb2


def get_scales_data(hist_coll,
                    light_graph,
                    convert_to_calib_graph,
                    hw_specs,
                    sw_config,
                    sim_params):
    # Get opu nodes
    filt = apply_node_map.ApplyNodeMap.get_opu_node_filter(sw_config)
    opu_nodes = [node for node in light_graph.nodes() if filt.matches(node, light_graph)]

    # Use hist_coll for scales that are not for padding
    precisions = calibration_pb2.SimpleMap()
    for opu_node in opu_nodes:
        keys = convert_to_calib_graph.node_name_to_keys(opu_node.name)
        for key in keys:
            if hist_coll.get_histogram_mode(key) != calibration_pb2.HM_PADDING:
                precisions.simple_map[key] = hw_specs.output_precision

    scales = hist_coll.get_quant_scales(sw_config.adc_scale_quantization_method,
                                        sw_config,
                                        precisions,
                                        common_pb2.QB_NONE)

    # Add scales for padding
    for opu_node in opu_nodes:
        keys = convert_to_calib_graph.node_name_to_keys(opu_node.name)
        for key in keys:
            if hist_coll.get_histogram_mode(key) == calibration_pb2.HM_PADDING:
                scales.scale_info_map[
                    key].scale = -1  # Use negative scales to tag paddings

    # Convert scales to QuantScalesData
    adc_scales_data = calibration_pb2.QuantScalesData()
    for opu_node in opu_nodes:
        keys = convert_to_calib_graph.node_name_to_keys(opu_node.name)

        node_scale_pair = adc_scales_data.data.add()
        node_scale_pair.node_info.node_name = opu_node.name
        node_scale_pair.node_info.quant_precision = int(precisions.simple_map[keys[0]])

        for index, key in enumerate(keys):
            if hist_coll.get_histogram_mode(key) != calibration_pb2.HM_PADDING:
                assert (precisions.simple_map[key] ==
                        node_scale_pair.node_info.quant_precision)

            node_scale_pair.scale_info_list.l.add().CopyFrom(scales.scale_info_map[key])

    return adc_scales_data


def get_virtual_sim_params(sim_params):
    # Update sim params
    virtual_sim_params = sim_params_pb2.SimulationParams()
    virtual_sim_params.CopyFrom(sim_params)
    virtual_sim_params.arch_params.arch_type = sim_params_pb2.ArchitectureParams.VIRTUAL

    return virtual_sim_params


def main(light_graph, calibration_data, hw_specs, sw_config, sim_params):
    """
    Do quantization calibration on the given light_graph

    Params:
        light_graph: graph that has already been quantized and transformed to
            use custom ops
        calibration_data: a inference_pb2.BatchedInferenceInput() protobufs
        hw_specs: a hw_spec_pb2.HardwareSpecs() protobuf
        sw_config: a sw_config_pb2.SoftwareConfig() protobuf
        sim_params: a sim_params_pb2.SimulationParams() protobuf

    Returns:
        A tuple (light_graph, max_graph, populate_graph) where
        light_graph that has optimal adc scales, populate_graph is None
        if using QM_MAX_ABS_VAL for adc scale calibration
    """

    # Collect histograms for adc scales
    with graph_collection.GraphCollection() as graph_coll:
        # Create calibration graph
        hist_coll = graph_coll.histogram_collection()
        convert_to_calib_graph = (
            convert_to_adc_scale_calibration_graph.ConvertToADCScaleCalibrationGraph(
                hw_specs,
                sw_config,
                sim_params,
                hist_coll))
        calib_graph = convert_to_calib_graph.process_transforms(light_graph)

        runner = histogram_graph_runner.HistogramGraphRunner(
            calib_graph,
            hw_specs,
            sw_config,
            get_virtual_sim_params(sim_params),
            graph_coll)
        runner.run(calibration_data)

        # Get scales data
        logging.info("-Computing Scales")
        adc_scales_data = get_scales_data(hist_coll,
                                          light_graph,
                                          convert_to_calib_graph,
                                          hw_specs,
                                          sw_config,
                                          sim_params)

    return adc_scales_data, calib_graph
