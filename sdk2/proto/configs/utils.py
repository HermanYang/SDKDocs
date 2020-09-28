from google.protobuf import text_format

from sdk2.proto import hw_spec_pb2, sim_params_pb2, sw_config_pb2


def _read_text_format(cls, path):
    proto = cls()
    with open(path, "r") as f:
        text_format.Parse(f.read(), proto)
    return proto


def _write_text_format(proto, path):
    with open(path, "w") as f:
        f.write(text_format.MessageToString(proto))


def read_hw_specs(hw_specs_path):
    return _read_text_format(hw_spec_pb2.HardwareSpecs, hw_specs_path)


def write_hw_specs(hw_specs, hw_specs_path):
    _write_text_format(hw_specs, hw_specs_path)


def read_sw_config(sw_config_path):
    return _read_text_format(sw_config_pb2.SoftwareConfig, sw_config_path)


def write_sw_config(sw_config, sw_config_path):
    _write_text_format(sw_config, sw_config_path)


def read_sim_params(sim_params_path):
    return _read_text_format(sim_params_pb2.SimulationParams, sim_params_path)


def write_sim_params(sim_params, sim_params_path):
    _write_text_format(sim_params, sim_params_path)
