from lt_sdk.proto import performance_data_pb2
from lt_sdk.proto.configs import (
    generate_hw_specs,
    generate_sim_params,
    generate_sw_config,
    param_sweep,
)


def create_sweep_dict(base_fn, *args, **kwargs):
    return {
        name: obj for obj,
        name in param_sweep.ParamSweep("",
                                       base_fn,
                                       *args,
                                       **kwargs).generate()
    }


def get_configs(perf_sweep=None, graph_type=None, nodes_filter=None, extra_mods_fn=None):
    """This is static, but also has an optional self to call extra_modifications."""
    hw_specs_dict = hw_specs_sweep_dict()
    if not nodes_filter and perf_sweep:
        nodes_filter = perf_sweep.ignore_nodes_filter()
    if graph_type is None and perf_sweep:
        graph_type = perf_sweep.graph_type()
    sw_config_dict = sw_config_sweep_dict(graph_type, ignore_nodes_filter=[nodes_filter])
    sim_params_dict = sim_params_sweep_dict()

    configs = []
    for hw_specs_name, hw_specs in hw_specs_dict.items():
        for sw_config_name, sw_config in sw_config_dict.items():
            for sim_params_name, sim_params in sim_params_dict.items():
                desc = "{}__{}__{}".format(hw_specs_name,
                                           sim_params_name,
                                           sw_config_name)
                if extra_mods_fn:
                    extra_mods_fn(hw_specs, sw_config, sim_params)

                config = performance_data_pb2.ConfigInfo()
                config.do_transform = True
                config.description = desc
                config.hw_specs.CopyFrom(hw_specs)
                config.sw_config.CopyFrom(sw_config)
                config.sim_params.CopyFrom(sim_params)

                configs.append(config)

    return configs


# ----------------- Modify kwargs here to specify what configs to run -------------------


def hw_specs_sweep_dict(*args):
    return create_sweep_dict(
        generate_hw_specs.generate_mosaic_delta,
        *args,
    )


def sw_config_sweep_dict(*args, **kwargs):
    return create_sweep_dict(
        generate_sw_config.generate_mosaic_delta,
        *args,
        **kwargs,
    )


def sim_params_sweep_dict(*args):
    return create_sweep_dict(
        generate_sim_params.generate_mosaic_delta,
        *args,
    )


# --------------------------------------------------------------------------------------------------
