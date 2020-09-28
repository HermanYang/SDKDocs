import argparse
import os

from tensorflow.core.framework import step_stats_pb2
from tensorflow.python.client import timeline

from sdk2.proto import inference_pb2, lgf_pb2, performance_data_pb2
from sdk2.proto.configs import generate_hw_specs, generate_sim_params, utils

# timeline label is: name = op(arg, arg, ...)
TIMELINE_LABEL_FORMAT = "{0} = {1}({2})"

INSTRUCTION_TRACE = "instructions.trace"
UMEM_TRACE = "umem.trace"
UARCH_TRACE = "uarch.trace"

ITX = "ITX"
DRAM = "DRAM"
IO_PORT = "IO PORT"


def get_instr_name(opu_inst):
    return opu_inst.node.WhichOneof("node")


def clocks_to_nanos(clks, sim_params):
    return int(1e3 * clks / (sim_params.arch_params.clock_frequency))


def _add_uarch_nodes(sim_params, devices, inst, op_name):
    for unit in inst.unit_name:
        dev = devices[unit]
        node = dev.node_stats.add()

        node.node_name = "[{0}] - {1}".format(inst.pc, op_name)
        node.timeline_label = TIMELINE_LABEL_FORMAT.format(node.node_name, op_name, "")
        node.all_start_micros = clocks_to_nanos(inst.start_clk, sim_params)
        node.all_end_rel_micros = clocks_to_nanos(inst.duration_clks, sim_params)


def _add_resource_nodes(sim_params, resources, inst, units_used):
    for name, units in units_used.items():
        for channel in units:
            dev = resources[name][channel]
            node = dev.node_stats.add()

            node.node_name = "[{0}] - {1}".format(inst.pc, "USED")
            node.timeline_label = TIMELINE_LABEL_FORMAT.format(
                node.node_name, "USED", "")
            node.all_start_micros = clocks_to_nanos(inst.start_clk, sim_params)
            node.all_end_rel_micros = clocks_to_nanos(inst.duration_clks, sim_params)


def _create_resources(step_stats, num_units_map):
    resources = {}
    for name, num_units in num_units_map.items():
        devices = resources.setdefault(name, [])
        for i in range(num_units):
            dev_stats = step_stats.dev_stats.add()
            devices.append(dev_stats)
            dev_stats.device = "{} {}".format(name, i)

    return resources


def instruction_trace(tracepath, sim_result, hw_specs, sim_params):
    step_stats = step_stats_pb2.StepStats()
    opu_dev_stats = step_stats.dev_stats.add()
    opu_dev_stats.device = "INSTRUCTIONS - OPU"
    cpu_dev_stats = step_stats.dev_stats.add()
    cpu_dev_stats.device = "INSTRUCTIONS - CPU"

    resources = _create_resources(
        step_stats, {
            ITX: sim_params.arch_params.num_rings,
            DRAM: sim_params.arch_params.num_memory_channels,
            IO_PORT: sim_params.arch_params.num_io_ports,
        })

    uarch_devices = {}

    pc_to_instr = {i.instruction.pc: i.instruction for i in sim_result.instructions}

    for i, inst in enumerate(sim_result.instructions):
        _add_resource_nodes(sim_params, resources, inst, {
            ITX: inst.interconnects,
            DRAM: inst.dram_channels,
            IO_PORT: inst.io_ports,
        })

        if inst.pc not in pc_to_instr:
            print(inst)
            print(pc_to_instr)
            raise ValueError("Something wrong: {0} {1} ".format(tracepath, inst))
        name = get_instr_name(pc_to_instr[inst.pc])

        node = opu_dev_stats.node_stats.add()
        op_name = name

        if pc_to_instr[inst.pc].node.HasField(lgf_pb2.LNF.bundle.DESCRIPTOR.name):
            op_name += (" [" + ", ".join(
                n.WhichOneof("node")
                for n in pc_to_instr[inst.pc].node.bundle.subgraph.nodes) + "]")

        for unit in inst.unit_name:
            if unit not in uarch_devices:
                instr_dev_stats = step_stats.dev_stats.add()
                instr_dev_stats.device = "UARCH - " + unit
                uarch_devices[unit] = instr_dev_stats
        _add_uarch_nodes(sim_params, uarch_devices, inst, name)

        try:
            tensor_name = pc_to_instr[inst.pc].tensor_name or pc_to_instr[
                inst.pc].dest_addr[0].info.name
        except IndexError:
            tensor_name = "unknown"

        dep_instr_names = []
        for dep_pc_diff in pc_to_instr[inst.pc].dependent_pcs_distance:
            dep_pc = inst.pc - dep_pc_diff
            if dep_pc in pc_to_instr:
                dep_instr_names.append(get_instr_name(pc_to_instr[dep_pc]))

        node.node_name = "[{0}] - {1}".format(inst.pc, tensor_name)
        node.timeline_label = TIMELINE_LABEL_FORMAT.format(node.node_name, op_name,
                                                           ", ".join(dep_instr_names))
        node.all_start_micros = clocks_to_nanos(inst.start_clk, sim_params)
        node.all_end_rel_micros = clocks_to_nanos(inst.duration_clks, sim_params)

    trace = timeline.Timeline(step_stats=step_stats)
    with open(tracepath, "w") as f:
        f.write(trace.generate_chrome_trace_format())

    return trace


def get_trace_file_name(user_path, tracedir):
    d = tracedir or os.path.dirname(user_path)
    fname = os.path.basename(user_path)
    if "." in user_path:
        fname = "".join(fname.split(".")[:-1] + [".trace"])
    else:
        fname = fname + ".trace"
    return os.path.join(d, fname)


def main(args):
    if args.spec_path:
        hw_specs = utils.read_hw_specs(args.spec_path)
        sim_params = utils.read_sim_params(args.simparams_path)
    elif args.spec_name:
        hw_specs = generate_hw_specs.CONFIGS[args.spec_name]()
        sim_params = generate_sim_params.CONFIGS[args.simparams_name]()

    sim_result = inference_pb2.ExecutionStats()
    if args.sim_result_path:
        with open(args.sim_result_path, "rb") as f:
            sim_result.ParseFromString(f.read())
        trace_path = get_trace_file_name(args.sim_result_path, args.tracedir)
    elif args.perf_data_path:
        perf_data = performance_data_pb2.PerformanceData()
        with open(args.perf_data_path, "rb") as f:
            perf_data.ParseFromString(f.read())
        sim_result.CopyFrom(perf_data.execution_stats)
        hw_specs = perf_data.config.hw_specs
        sim_params = perf_data.config.sim_params
        trace_path = get_trace_file_name(args.perf_data_path, args.tracedir)
    else:
        raise ValueError("Must specify either sim_result_path or perf_data_path")

    instruction_trace(trace_path, sim_result, hw_specs, sim_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_result_path",
                        default=None,
                        type=str,
                        help="path to serialized SimulationResult proto.")
    parser.add_argument("--perf_data_path",
                        default=None,
                        type=str,
                        help="path to serialized PerformanceData proto.")
    parser.add_argument("--tracedir",
                        default=None,
                        type=str,
                        help="Dir to put outputs in chrome tracing format.")

    # Specify either paths or names for specs and params.
    parser.add_argument("--spec_path",
                        default=None,
                        type=str,
                        help="path to spec that was run on.")
    parser.add_argument("--simparams_path",
                        default=None,
                        type=str,
                        help="path to sim params that was run on.")

    parser.add_argument("--spec_name",
                        default=None,
                        type=str,
                        help="Name of spec that was run on.")
    parser.add_argument("--simparams_name",
                        default=None,
                        type=str,
                        help="Name of sim params that was run on.")

    args = parser.parse_args()
    main(args)
