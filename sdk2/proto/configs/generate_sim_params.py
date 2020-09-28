import argparse
import math
import os

from sdk2.proto import sim_params_pb2
from sdk2.proto.configs import param_sweep, utils

ANALOG_LUT_DIR = "sdk2/simulation/opu_model/lut"


def _set_default_arch_params(sim_params,
                             clock_frequency=1000,
                             modulation_rate=1000,
                             num_memory_channels=4,
                             load_weight_ns=10,
                             weight_settling_ns=50,
                             opu_latency_clks=40,
                             arch_type=sim_params_pb2.ArchitectureParams.VIRTUAL,
                             issue_clocks=1,
                             num_io_ports=1):
    arch_params = sim_params.arch_params

    arch_params.clock_frequency = clock_frequency
    arch_params.modulation_rate = modulation_rate
    arch_params.memory_bandwidth = 64000000000
    arch_params.memory_latency_ns = 40
    arch_params.num_memory_channels = num_memory_channels

    arch_params.host_to_device_bandwidth = 32000000000
    arch_params.host_to_device_latency_ns = 1000
    arch_params.num_io_ports = num_io_ports
    arch_params.load_weights_ns = load_weight_ns
    arch_params.weight_settling_ns = weight_settling_ns
    arch_params.opu_latency_clks = opu_latency_clks
    arch_params.lookahead_window = 3
    arch_params.issue_clocks = issue_clocks
    arch_params.parallel_issue = True
    arch_params.umem_ports_per_bank = 1
    arch_params.load_weights_input_ports = 8
    arch_params.uarch = sim_params_pb2.ArchitectureParams.HETERO_INSTR
    arch_params.interconnect = sim_params_pb2.ArchitectureParams.ITX_RING
    arch_params.num_rings = 4
    arch_params.num_uarch_units = 8
    arch_params.arch_type = arch_type
    arch_params.add_dequant_bias_before_accumulate = False
    arch_params.pack_hostmem = True


def _set_default_power_model(sim_params):
    power = sim_params.power

    power.opu_tx_pj = 6.0
    power.opu_rx_pj = 6.0
    power.opu_adc_pj = 7.0
    power.opu_dac_pj = 5.5
    power.sram_read_pj = 1.0
    power.sram_write_pj = 1.0
    power.alu_op_pj = 2.0
    power.dram_read_pj = 11.0
    power.dram_write_pj = 11.0
    power.onchip_interconnect_pj = 1.5

    power.wx_voltage = 1.8
    power.wx_overhead_ma = 2.0
    power.wx_lp_max_ma = 1.0
    power.wx_hp_max_ma = 5.0
    power.wx_cutoff_value = 32

    power.laser_w = 32
    power.misc_power_w = 20


def _set_default_analog_specs(sim_params,
                              adc_noise_factor=0,
                              tia_noise_factor=0,
                              data_noise_factor=0,
                              weight_noise_factor=0,
                              enable_pd_noise=False,
                              fab_variation=0,
                              expected_tia_signal_width=0.2,
                              laser_power_per_col=100e-6,
                              data_lut_file="SCIM_ALGO_cutoff+-3_wER.csv",
                              weight_lut_file="CIMZI_ALGO_1to0.csv"):
    analog_params = sim_params.analog_params

    analog_params.adc_ref_volt = 2
    analog_params.adc_noise_factor = adc_noise_factor
    analog_params.tia_supply_volt = 2
    analog_params.tia_bias_volt = 1
    analog_params.tia_noise_factor = tia_noise_factor
    analog_params.pd_conversion_factor = 1
    analog_params.data_noise_factor = data_noise_factor
    analog_params.weight_noise_factor = weight_noise_factor
    analog_params.enable_pd_noise = enable_pd_noise
    analog_params.fab_variation = fab_variation
    analog_params.weight_dac_res = 8

    # Default laser power per weight column. Each column has 64 rows
    dimension = 64
    # Laser power per SCIM
    analog_params.laser_power = laser_power_per_col / dimension
    analog_params.tia_resistance = (0.5 * analog_params.tia_supply_volt /
                                    laser_power_per_col /
                                    analog_params.pd_conversion_factor)
    analog_params.data_lut_file = os.path.join(ANALOG_LUT_DIR, data_lut_file)
    analog_params.weight_lut_file = os.path.join(ANALOG_LUT_DIR, weight_lut_file)

    # Calculate the laser power required to make the width of differentiated
    # input currents of TIA, after multiplying the TIA resistance, equals
    # expected_tia_signal_width * tia_supply_volt.
    # This calculation assumes uniformly distributed unsigned inputs and uniformed
    # distributed signed weights. Under this circumstance, one can show that the
    # total differentiated optical power (positive branch - negative branch)
    # coming out of an OPU column averages at zero, with a width equal to the total
    # laser power coming into one OPU column divided by (3 * sqrt(dimension)).
    # NB: This is expected laser power per SCIM, to be consistent with
    # analog_params.laser_power
    analog_params.expected_laser_power = (expected_tia_signal_width * 3 *
                                          math.sqrt(dimension) *
                                          analog_params.tia_supply_volt /
                                          analog_params.tia_resistance /
                                          analog_params.pd_conversion_factor / dimension)


def generate_mosaic(clock_frequency=1000,
                    num_memory_channels=4,
                    compiled_batch_size=1,
                    num_runtime_threads=32,
                    adc_noise_factor=0,
                    tia_noise_factor=0,
                    data_noise_factor=0,
                    weight_noise_factor=0,
                    enable_pd_noise=False,
                    fab_variation=0,
                    num_calib_measurements=25,
                    expected_tia_signal_width=0.2,
                    laser_power_per_col=100e-6,
                    data_lut_file="SCIM_veriloga.csv",
                    weight_lut_file="CIMZI_veriloga.csv",
                    load_weight_ns=10,
                    opu_latency_clks=40,
                    weight_settling_ns=50,
                    perf_only=False,
                    arch_type=sim_params_pb2.ArchitectureParams.VIRTUAL,
                    issue_clocks=1):
    sim_params = sim_params_pb2.SimulationParams()

    # Architecture params
    _set_default_arch_params(sim_params,
                             clock_frequency=clock_frequency,
                             modulation_rate=clock_frequency,
                             num_memory_channels=num_memory_channels,
                             load_weight_ns=load_weight_ns,
                             weight_settling_ns=weight_settling_ns,
                             arch_type=arch_type,
                             opu_latency_clks=opu_latency_clks,
                             issue_clocks=issue_clocks)

    # Power params
    _set_default_power_model(sim_params)

    # Analog parameters
    _set_default_analog_specs(sim_params,
                              adc_noise_factor=adc_noise_factor,
                              tia_noise_factor=tia_noise_factor,
                              data_noise_factor=data_noise_factor,
                              weight_noise_factor=weight_noise_factor,
                              enable_pd_noise=enable_pd_noise,
                              fab_variation=fab_variation,
                              expected_tia_signal_width=expected_tia_signal_width,
                              laser_power_per_col=laser_power_per_col,
                              data_lut_file=data_lut_file,
                              weight_lut_file=weight_lut_file)

    # Extra fields
    sim_params.compiled_batch_size = compiled_batch_size
    sim_params.num_runtime_threads = num_runtime_threads
    sim_params.num_calib_measurements = num_calib_measurements

    sim_params.perf_only = perf_only

    return sim_params


def generate_mosaic_bravo(umem_ports_per_bank=1, num_runtime_threads=4, **kwargs):
    sim_params = generate_mosaic(**kwargs)
    sim_params.arch_params.umem_ports_per_bank = umem_ports_per_bank
    sim_params.arch_params.interconnect = sim_params_pb2.ArchitectureParams.ITX_XBAR
    sim_params.arch_params.uarch = sim_params_pb2.ArchitectureParams.MOSAIC_BRAVO
    sim_params.num_runtime_threads = num_runtime_threads

    del sim_params.analog_params.tia_gains[:]

    sim_params.analog_params.tia_gains.extend([1, 0.5, 0.2])

    return sim_params


def set_default_usim_params(sim_params):
    sim_params.arch_params.usim_params.crossbar_latency = 1
    sim_params.arch_params.usim_params.core_to_broadcast_latency = 1
    sim_params.arch_params.usim_params.broadcast_to_core_latency = 1
    sim_params.arch_params.usim_params.core_rx_latency = 1
    sim_params.arch_params.usim_params.inst_iss_to_core_latency = 1
    sim_params.arch_params.usim_params.hop_latency = 1
    sim_params.arch_params.usim_params.internal_core_latency = 1
    sim_params.arch_params.usim_params.internal_network_transaction_size = 128


def generate_mosaic_delta(bit_error_rate=0, issue_clocks=20, **kwargs):
    sim_params = generate_mosaic(weight_settling_ns=0,
                                 load_weight_ns=8,
                                 opu_latency_clks=40,
                                 issue_clocks=issue_clocks,
                                 **kwargs)
    sim_params.analog_params.bit_error_rate = bit_error_rate

    if sim_params.arch_params.arch_type == sim_params_pb2.ArchitectureParams.USIM:
        # Need defaults of non-zero
        set_default_usim_params(sim_params)

    generate_perfsim_mosaic_delta(sim_params)
    return sim_params


def generate_perfsim_mosaic_delta(sim_params):
    perf_sim_params = sim_params.perf_params

    perf_sim_params.common.clock_frequency = 1000

    perf_sim_params.mosaic.issue_window_size = 3
    perf_sim_params.mosaic.issue_clocks = 10
    perf_sim_params.mosaic.apply_weights_clocks = 4
    perf_sim_params.mosaic.off_chip_move_latency = 10
    perf_sim_params.mosaic.per_core_io_bandwidth = 6
    perf_sim_params.mosaic.matmul_latency_clocks = 15

    perf_sim_params.mosaic.num_resources[sim_params_pb2.MosaicParams.RSC_MATMUL] = 1
    perf_sim_params.mosaic.num_resources[sim_params_pb2.MosaicParams.RSC_WSU] = 1
    perf_sim_params.mosaic.num_resources[sim_params_pb2.MosaicParams.RSC_ALU] = 3
    perf_sim_params.mosaic.num_resources[sim_params_pb2.MosaicParams.RSC_UMEM_RD] = 5
    perf_sim_params.mosaic.num_resources[sim_params_pb2.MosaicParams.RSC_UMEM_WR] = 5
    perf_sim_params.mosaic.num_resources[sim_params_pb2.MosaicParams.RSC_IO] = 1

    return perf_sim_params


CONFIG_MOSAIC = "mosaic"
CONFIGS = {CONFIG_MOSAIC: generate_mosaic}

SWEEP_MOSAIC = "mosaic"
SWEEPS = {
    SWEEP_MOSAIC:
        param_sweep.ParamSweep(SWEEP_MOSAIC,
                               generate_mosaic,
                               clock_frequency=[1000, 2000, 2500])
}


def generate(out_dir=None, config_name=CONFIG_MOSAIC, sweep_name=None):
    out_dir = out_dir or os.path.dirname(__file__)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if config_name:
        utils.write_sim_params(
            CONFIGS[config_name](),
            os.path.join(out_dir, "{0}.sim_params.pb.txt".format(config_name)))

    paths = {}
    if sweep_name:
        for obj, cfg_name in SWEEPS[sweep_name].generate():
            config_path = os.path.join(
                out_dir, "{0}_{1}.sim_params.pb.txt".format(sweep_name, cfg_name))
            utils.write_sim_params(obj, config_path)
            paths[sweep_name] = config_path

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name",
                        default=None,
                        type=str,
                        help="Name of sim params config.")
    parser.add_argument("--sweep_name",
                        default=None,
                        type=str,
                        help="Name of set of sim params.")
    parser.add_argument("--out_dir",
                        default=os.path.join(os.path.expanduser("~"), "sim_configs"),
                        type=str,
                        help="Name of set of sim params.")
    args = parser.parse_args()
    generate(out_dir=args.out_dir,
             config_name=args.config_name,
             sweep_name=args.sweep_name)
