from lt_sdk.proto import graph_types_pb2, hardware_configs_pb2
from lt_sdk.proto.configs import (
    generate_hw_specs,
    generate_sim_params,
    generate_sw_config,
)


def get_config(hw_cfg, graph_type):
    if hw_cfg == hardware_configs_pb2.DELTA:
        return (generate_hw_specs.generate_mosaic_delta(),
                generate_sw_config.generate_mosaic_delta(graph_type),
                generate_sim_params.generate_mosaic_delta())
    elif hw_cfg == hardware_configs_pb2.BRAVO:
        return (generate_hw_specs.generate_mosaic_bravo(),
                generate_sw_config.generate_standard_sw_config(graph_type),
                generate_sim_params.generate_mosaic_bravo())
    else:
        raise ValueError("Unknown hardware config: {0}".format(hw_cfg))


TYPE_MAP = {
    "lgf": graph_types_pb2.LGFProtobuf,
    "tfsm": graph_types_pb2.TFSavedModel,
    "tf": graph_types_pb2.TFSavedModel,
    "savedmodel": graph_types_pb2.TFSavedModel,
    "tflite": graph_types_pb2.TFLiteSavedModel,
    "onnx": graph_types_pb2.ONNXModel
}


def get_graph_type(type_name):
    return TYPE_MAP[type_name.lower()]
