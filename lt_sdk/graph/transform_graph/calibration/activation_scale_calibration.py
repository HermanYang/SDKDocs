import logging

from lt_sdk.graph.graph_collections import graph_collection
from lt_sdk.graph.run_graph import histogram_graph_runner
from lt_sdk.graph.transform_graph.calibration import (
    convert_to_activation_scale_calibration_graph,
)
from lt_sdk.proto import calibration_pb2


def get_scales_data(hist_coll, nodes_to_calibrate, convert_to_calib_graph, sw_config):
    # Create precisions map
    precisions = calibration_pb2.SimpleMap()
    precisions.simple_map.update({
        convert_to_calib_graph.get_key_from_node_name(n.node_name):
        n.quant_precision + sw_config.use_unsigned_quant_scheme
        for n in nodes_to_calibrate
    })

    # Compute the scales
    scales = hist_coll.get_quant_scales(
        sw_config.activation_scale_quantization_method,
        sw_config,
        precisions,
        sw_config.activation_scale_quantization_bias_type,
        sw_config.use_unsigned_quant_scheme)

    activation_scales_data = calibration_pb2.QuantScalesData()
    for node_info in nodes_to_calibrate:
        node_scale_pair = activation_scales_data.data.add()
        node_scale_pair.node_info.CopyFrom(node_info)
        node_scale_pair.scale_info.CopyFrom(
            scales.scale_info_map[convert_to_calib_graph.get_key_from_node_name(
                node_info.node_name)])

    return activation_scales_data


def main(light_graph,
         calibration_data,
         hw_specs,
         sw_config,
         sim_params,
         nodes_to_calibrate):
    """
    Do activation scale calibration on the given light_graph

    Params:
        light_graph: a LightGraph object that has not been transformed
        calibration_data: a inference_pb2.BatchedInferenceInput() protobufs
        hw_specs: a hw_spec_pb2.HardwareSpecs() protobuf
        sw_config: a sw_config_pb2.SoftwareConfig() protobuf
        sim_params: a sim_params_pb2.SimulationParams() protobuf
        nodes_to_calibrate: a list of calibration_pb2.NodeInfo() objects

    Returns:
        activation_scales_data: a calibration_pb2.QuantScalesData() protobuf
            containing quantization scales for the nodes in nodes_to_calibrate
    """
    with graph_collection.GraphCollection() as graph_coll:
        # Create calibration graph
        hist_coll = graph_coll.histogram_collection()
        convert_to_calib_graph = (convert_to_activation_scale_calibration_graph.
                                  ConvertToActivationScaleCalibrationGraph(
                                      nodes_to_calibrate,
                                      sw_config,
                                      hist_coll))
        calib_graph = convert_to_calib_graph.process_transforms(light_graph)

        runner = histogram_graph_runner.HistogramGraphRunner(calib_graph,
                                                             hw_specs,
                                                             sw_config,
                                                             sim_params,
                                                             graph_coll)
        runner.run(calibration_data)

        # Get scales data
        logging.info("-Computing Scales")
        activation_scales_data = get_scales_data(hist_coll,
                                                 nodes_to_calibrate,
                                                 convert_to_calib_graph,
                                                 sw_config)

    return activation_scales_data
