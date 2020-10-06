from dataclasses import dataclass

from lt_sdk.proto import (
    graph_types_pb2,
    hardware_configs_pb2,
    hw_spec_pb2,
    performance_data_pb2,
    sim_params_pb2,
    sw_config_pb2,
)
from lt_sdk.proto.configs import config as proto_config


@dataclass
class LightConfig:
    sw_config: sw_config_pb2.SoftwareConfig
    hw_specs: hw_spec_pb2.HardwareSpecs
    sim_params: sim_params_pb2.SimulationParams

    def get_configs(self):
        return self.sw_config, self.hw_specs, self.sim_params

    @classmethod
    def from_proto(cls, config: performance_data_pb2.ConfigInfo):
        return cls(sw_config=config.sw_config,
                   hw_specs=config.hw_specs,
                   sim_params=config.sim_params)

    def to_proto(self):
        config = performance_data_pb2.ConfigInfo()
        config.sw_config.CopyFrom(self.sw_config)
        config.hw_specs.CopyFrom(self.hw_specs)
        config.sim_params.CopyFrom(self.sim_params)

        return config


def get_default_config(hw_cfg=hardware_configs_pb2.DELTA,
                       graph_type=graph_types_pb2.TFSavedModel):
    hw_specs, sw_config, sim_params = proto_config.get_config(hw_cfg, graph_type)

    return LightConfig(sw_config=sw_config, hw_specs=hw_specs, sim_params=sim_params)
