import argparse
import logging
import os

from lt_sdk.common import py_test_util
from lt_sdk.graph import lgf_graph
from lt_sdk.graph.transform_graph.graph_transformers import collapse_supported_subgraphs
from lt_sdk.perfsim import mosaic, perfsim_logging
from lt_sdk.proto import graph_types_pb2, lgf_pb2, performance_data_pb2, sim_params_pb2
from lt_sdk.runtime import compiler
from lt_sdk.verification import performance_sweep, run_performance_sweep_params
from lt_sdk.visuals import sim_result_to_trace

MODEL_CLASSES = {
    sim_params_pb2.PerfSimParams.mosaic.DESCRIPTOR.name: mosaic.MosaicModel,
}


def compile_subgraphs(lg, cfg):
    collapsed = collapse_supported_subgraphs.CollapseSupportedSubgraphs().\
            process_transforms(lg)

    ret = []
    for n in collapsed.nodes():
        if n.HasField(lgf_pb2.LNF.subgraph.DESCRIPTOR.name):
            opu_bin = compiler.compile_subgraph(n.subgraph.graph,
                                                cfg.hw_specs,
                                                cfg.sw_config,
                                                cfg.sim_params)
            ret.append(opu_bin)
    return ret


def simulate_subgraph(subgraph_bin, perf_data):
    perf = perf_data.config.sim_params.perf_params
    model_cls = MODEL_CLASSES[perf.WhichOneof("model_class")](perf_data.config)
    model_cls.simulate(subgraph_bin, perf_data)


def simulate(graph, config):
    perf_data = performance_data_pb2.PerformanceData()
    perf_data.config.CopyFrom(config)

    subgraph_binaries = compile_subgraphs(graph, config)
    for i, sub in enumerate(subgraph_binaries):
        perfsim_logging.debug("-- Subgraph {0}: {1}".format(i, len(sub.instr)))
        simulate_subgraph(sub, perf_data)

    return perf_data


def main(graph_path, output_dir):
    perfsim_logging.LEVELS_ENABLED.add(perfsim_logging.LogLevel.ISSUE)
    perfsim_logging.LEVELS_ENABLED.add(perfsim_logging.LogLevel.COMPLETE)
    perfsim_logging.LEVELS_ENABLED.add(perfsim_logging.LogLevel.ISSUE_WINDOW)
    perfsim_logging.LEVELS_ENABLED.add(perfsim_logging.LogLevel.HAZARDS)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    lg = lgf_graph.LightGraph.from_pb(graph_path)

    configs = run_performance_sweep_params.get_configs(
        graph_type=graph_types_pb2.LGFProtobuf)

    sweep_data = performance_data_pb2.PerformanceSweepData()
    for i, cfg in enumerate(configs):
        perfsim_logging.debug("------- Running config {0}: {1} ---------".format(
            i,
            cfg.description))
        perf_data = sweep_data.data.add()
        perf_data.CopyFrom(simulate(lg, cfg))

        with open(os.path.join("performance_data_{}.pb".format(i)), "wb") as f:
            f.write(perf_data.SerializeToString())

        sim_result_to_trace.instruction_trace(
            os.path.join(output_dir,
                         "performance_data_{}.trace".format(i)),
            perf_data.execution_stats,
            cfg.hw_specs,
            cfg.sim_params)

    with open(os.path.join(output_dir, performance_sweep.SWEEP_NAME), "wb") as f:
        f.write(sweep_data.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_path", type=str, help="path to LGF proto")
    parser.add_argument("--output_dir", type=str, help="Dir to drop results")

    args = parser.parse_args()

    py_test_util.PythonTestProgram.set_root_logger(logging_level=logging.INFO,
                                                   logging_format="%(message)s")

    main(args.graph_path, args.output_dir)
