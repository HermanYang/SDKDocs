import argparse
import math
import os

import numpy as np

from lt_sdk.proto import hw_spec_pb2, subgraph_binary_pb2
from lt_sdk.proto.configs import param_sweep, utils


class Precision(object):

    def __init__(self, precision, signed, time_multiplex):
        self.precision = precision
        if precision > 0:
            self.signed = signed
            self.time_multiplex = time_multiplex
        else:  # Unquantized data
            self.signed = True
            self.time_multiplex = False

    def __str__(self):
        res = [str(self.precision), "SIGNED" if self.signed else "UNSIGNED"]
        if self.time_multiplex:
            res.append("TDM")
        return "_".join(res)


class InputPrecision(Precision):

    def __init__(self, precision=4, signed=False, time_multiplex=True):
        super().__init__(precision, signed, time_multiplex)


class PhasePrecision(Precision):

    def __init__(self, precision=6, signed=True, time_multiplex=False):
        super().__init__(precision, signed, time_multiplex)


def get_num_columns(hw_specs):
    if hw_specs.HasField(hw_spec_pb2.HardwareSpecs.ideal_incoherent.DESCRIPTOR.name):
        return hw_specs.ideal_incoherent.output_columns
    elif hw_specs.HasField(
            hw_spec_pb2.HardwareSpecs.physical_incoherent.DESCRIPTOR.name):
        return hw_specs.physical_incoherent.output_columns
    elif hw_specs.HasField(hw_spec_pb2.HardwareSpecs.ideal_digital.DESCRIPTOR.name):
        return hw_specs.ideal_digital.output_columns
    elif hw_specs.HasField(hw_spec_pb2.HardwareSpecs.physical_digital.DESCRIPTOR.name):
        return hw_specs.physical_digital.output_columns
    elif hw_specs.HasField(hw_spec_pb2.HardwareSpecs.coherent.DESCRIPTOR.name):
        return hw_specs.coherent.capacity
    else:
        raise ValueError("Invalid hardware specs")


def generate_mosaic(input_dim=64,
                    input_precision=InputPrecision(),
                    phase_precision=PhasePrecision(),
                    output_precision=8,
                    use_ideal_mode=True,
                    value_storage_bits=16,
                    pseudo_energy_precision=6):
    hw_specs = hw_spec_pb2.HardwareSpecs()

    if use_ideal_mode:
        hw_specs.ideal_incoherent.SetInParent()
        incoherent_specs = hw_specs.ideal_incoherent
    if not use_ideal_mode:
        assert (not input_precision.signed)
        assert (not phase_precision.signed or phase_precision.precision <= 0)
        hw_specs.physical_incoherent.SetInParent()
        incoherent_specs = hw_specs.physical_incoherent

    hw_specs.dimension = input_dim
    incoherent_specs.output_columns = input_dim

    hw_specs.input_precision = input_precision.precision
    hw_specs.signed_inputs = input_precision.signed
    hw_specs.time_multiplex_inputs = input_precision.time_multiplex

    if phase_precision.precision > 0:
        hw_specs.use_unquantized_weights = False
        hw_specs.phase_precision = phase_precision.precision
    else:
        hw_specs.use_unquantized_weights = True
        hw_specs.phase_precision = 32
    hw_specs.signed_phases = phase_precision.signed
    hw_specs.time_multiplex_weights = phase_precision.time_multiplex

    hw_specs.max_abs_weight = 1 / np.sqrt(2)

    hw_specs.num_opus = 1

    hw_specs.output_precision = output_precision
    incoherent_specs.pseudo_energy_precision = pseudo_energy_precision

    hw_specs.tmem_rows = 16777216
    hw_specs.umem_rows_per_bank = 65536
    hw_specs.umem_num_banks = 8
    hw_specs.const_mem_rows = 28000
    hw_specs.adc_scale_mem_rows = 50000
    hw_specs.accumulators_mem_rows = 75000

    hw_specs.value_storage_bits = value_storage_bits

    hw_specs.num_wavelengths = 1
    hw_specs.weight_memory = subgraph_binary_pb2.MemoryAllocation.UMEM
    hw_specs.constant_memory = subgraph_binary_pb2.MemoryAllocation.UMEM

    return hw_specs


def _set_resnet_required_memories(
        hw_specs,
        num_umem_banks=16,
        weight_memory=subgraph_binary_pb2.MemoryAllocation.UMEM,
        constant_memory=subgraph_binary_pb2.MemoryAllocation.UMEM):
    hw_specs.tmem_rows = 128330
    hw_specs.const_mem_rows = 0

    umem_total_rows = 202544
    hw_specs.umem_num_banks = num_umem_banks
    hw_specs.umem_rows_per_bank = math.ceil(umem_total_rows / hw_specs.umem_num_banks)

    hw_specs.adc_scale_mem_rows = 2256
    hw_specs.accumulators_mem_rows = 6272

    hw_specs.weight_memory = weight_memory
    hw_specs.constant_memory = constant_memory


def generate_mosaic_bravo(num_umem_banks=3, num_wavelengths=4, use_ideal_mode=True):
    hw_specs = generate_mosaic(
        input_dim=64,
        input_precision=InputPrecision(precision=4,
                                       signed=False,
                                       time_multiplex=True),
        phase_precision=PhasePrecision(precision=6,
                                       signed=False,
                                       time_multiplex=False),
        output_precision=8,
        use_ideal_mode=use_ideal_mode,
    )

    hw_specs.num_wavelengths = num_wavelengths
    _set_resnet_required_memories(hw_specs, num_umem_banks=num_umem_banks)

    return hw_specs


def generate_mosaic_delta(use_ideal_mode=True,
                          dimension=128,
                          output_columns=128 * 3,
                          num_umem_banks=16,
                          num_opus=1,
                          weight_memory=subgraph_binary_pb2.MemoryAllocation.UMEM,
                          constant_memory=subgraph_binary_pb2.MemoryAllocation.UMEM):
    hw_specs = hw_spec_pb2.HardwareSpecs()

    if use_ideal_mode:
        hw_specs.ideal_digital.SetInParent()
        digital_specs = hw_specs.ideal_digital
    if not use_ideal_mode:
        hw_specs.physical_digital.SetInParent()
        digital_specs = hw_specs.physical_digital

    hw_specs.input_precision = 8
    hw_specs.signed_inputs = True
    hw_specs.time_multiplex_inputs = False

    hw_specs.phase_precision = 8
    hw_specs.signed_phases = True
    hw_specs.time_multiplex_weights = False

    hw_specs.output_precision = 23
    hw_specs.value_storage_bits = 16
    digital_specs.output_columns = output_columns
    hw_specs.dimension = dimension
    hw_specs.num_opus = num_opus

    digital_specs.multiplication_precision = 2 * hw_specs.input_precision
    digital_specs.sum_tree_precision.extend([
        digital_specs.multiplication_precision + i + 1
        for i in range(np.log2(hw_specs.dimension).astype(int))
    ])

    hw_specs.num_wavelengths = 1

    _set_resnet_required_memories(hw_specs,
                                  num_umem_banks=num_umem_banks,
                                  weight_memory=weight_memory,
                                  constant_memory=constant_memory)

    return hw_specs


def generate_vanguard(**kwargs):
    hw_specs = generate_mosaic_delta(**kwargs)

    hw_specs.tmem_rows = 128330
    hw_specs.const_mem_rows = 0

    umem_total_rows = 428000
    hw_specs.umem_num_banks = 32
    hw_specs.umem_rows_per_bank = math.ceil(umem_total_rows / hw_specs.umem_num_banks)

    hw_specs.adc_scale_mem_rows = 6300
    hw_specs.accumulators_mem_rows = 6300 * 5

    return hw_specs


CONFIG_MOSAIC_BRAVO_IDEAL = "mosaic_bravo_ideal"
CONFIG_MOSAIC_BRAVO_PHYSICAL = "mosaic_bravo_physical"
CONFIG_MOSAIC_DELTA_IDEAL = "mosaic_delta_ideal"
CONFIG_MOSAIC_DELTA_PHYSICAL = "mosaic_delta_physical"
CONFIGS = {
    CONFIG_MOSAIC_BRAVO_IDEAL: (generate_mosaic_bravo,
                                {
                                    "use_ideal_mode": True
                                }),
    CONFIG_MOSAIC_BRAVO_PHYSICAL: (generate_mosaic_bravo,
                                   {
                                       "use_ideal_mode": False
                                   }),
    CONFIG_MOSAIC_DELTA_IDEAL: (generate_mosaic_delta,
                                {
                                    "use_ideal_mode": True
                                }),
    CONFIG_MOSAIC_DELTA_PHYSICAL: (generate_mosaic_delta,
                                   {
                                       "use_ideal_mode": False
                                   })
}

SWEEP_MOSAIC = "mosaic"
SWEEPS = {
    SWEEP_MOSAIC:
        param_sweep.ParamSweep(SWEEP_MOSAIC,
                               generate_mosaic,
                               input_dim=[16,
                                          32,
                                          64])
}


def generate(out_dir=None, config_name=CONFIG_MOSAIC_BRAVO_IDEAL, sweep_name=None):
    out_dir = out_dir or os.path.dirname(__file__)

    print("writing to: {0}".format(out_dir))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if config_name:
        func, kwargs = CONFIGS[config_name]
        utils.write_hw_specs(func(**kwargs),
                             os.path.join(out_dir,
                                          "{0}.pb.txt".format(config_name)))

    paths = {}
    if sweep_name:
        for obj, cfg_name in SWEEPS[sweep_name].generate():
            config_path = os.path.join(out_dir,
                                       "{0}_{1}.pb.txt".format(sweep_name,
                                                               cfg_name))
            utils.write_hw_specs(obj, config_path)
            paths[sweep_name] = config_path

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name",
                        default=None,
                        type=str,
                        help="Name of hardware config.")
    parser.add_argument("--sweep_name",
                        default=None,
                        type=str,
                        help="Name of set of hardware configs.")
    parser.add_argument("--out_dir",
                        default=os.path.join(os.path.expanduser("~"),
                                             "configs"),
                        type=str,
                        help="Name of set of hardware configs.")
    args = parser.parse_args()
    generate(out_dir=args.out_dir,
             config_name=args.config_name,
             sweep_name=args.sweep_name)
