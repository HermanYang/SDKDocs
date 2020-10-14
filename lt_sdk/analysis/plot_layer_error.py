import argparse
import logging
import os

import numpy as np

from lt_sdk.common import py_graph_test_util, py_test_util
from lt_sdk.graph import full_graph_pipeline, lgf_graph
from lt_sdk.graph.run_graph import graph_runner
from lt_sdk.graph.transform_graph import utils
from lt_sdk.graph.transform_graph.graph_transformers import apply_node_map
from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    electronic_op_transform,
)
from lt_sdk.proto import (
    graph_types_pb2,
    hardware_configs_pb2,
    lgf_pb2,
    node_filters,
    ops_pb2,
    performance_data_pb2,
)
from lt_sdk.proto.configs import config
from lt_sdk.verification import performance_sweep_map


def _is_opu(node):
    assert (node.HasField(lgf_pb2.LNF.original.DESCRIPTOR.name))
    return node.original.op in [ops_pb2.MATMUL, ops_pb2.CONV2D, ops_pb2.DEPTHWISE_CONV2D]


def _get_opu_layer_graph(lgf, cal_data, max_nodes=32, opu_only=False):
    """Return a graph with all the opu ops as output edges, in execution-order."""
    node_filter = node_filters.and_filter(
        node_filters.not_filter(node_filters.op_filter(ops_pb2.UNKNOWN)),
        node_filters.not_filter(node_filters.op_filter(ops_pb2.CONST)))

    for output_edge in lgf.output_edges():
        output_node = lgf.get_node_by_name(output_edge.name)
        if node_filter.matches(output_node, lgf):
            ordered = list(lgf.bfs(output_node, node_filter=node_filter))
            break

    ordered.reverse()
    opu_edges = []

    for i, node_obj in enumerate(ordered):
        if not opu_only or _is_opu(node_obj):
            opu_edges.append(node_obj.outputs[0])
            if len(opu_edges) > max_nodes:
                break

    if not opu_edges:
        raise ValueError("Could not find OPU node")

    return opu_edges


def _get_outputs(perf_sweep, runner, sw_config):
    shard_outs = []
    for shard_indx in range(perf_sweep.num_test_shards()):
        logging.info("-Running inference on test data shard {}".format(shard_indx))
        test_inputs = perf_sweep.get_test_inputs(sw_config, shard_indx)
        shard_outs.append(runner.run(test_inputs))
    return shard_outs


class BNOffsetTransform(electronic_op_transform.ElectronicOpTransform):
    """Counts batch norm constants."""

    def __init__(self, corrections, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._corrections = corrections

    def transform(self, offset_node, light_graph):
        """
        Apply mean correction.
        """
        if offset_node.name not in self._corrections:
            raise ValueError("No correction for node {0}".format(offset_node.name))

        orig_arr = utils.tensor_pb_to_array(offset_node.const.value, np.float32)
        new_arr = orig_arr - self._corrections[offset_node.name]
        offset_node.const.value.CopyFrom(
            utils.array_to_tensor_pb(new_arr,
                                     offset_node.const.value.dtype))
        return self.create_transform_result(to_replace=[offset_node])


def collect_data_dict(data_d, shard_outs):
    for batch in shard_outs.batches:
        for named_tensor in batch.results:
            data_d.setdefault(named_tensor.edge_info.name,
                              []).append(
                                  utils.tensor_pb_to_array(named_tensor.data,
                                                           np.float32))


def get_layer_errors(workload,
                     output_dir,
                     hw_cfg,
                     opu_only=False,
                     max_nodes=32,
                     fix_bns=True):
    num_threads = 64
    perf_sweep = performance_sweep_map.get_sweep(workload, output_dir)

    hw_specs, sw_config, sim_params = config.get_config(hw_cfg, perf_sweep.graph_type())
    sw_config.num_threads_scales = num_threads
    sim_params.num_runtime_threads = num_threads

    sw_config.sweep_info.py_batch_size = perf_sweep.py_batch_size()
    sw_config.sweep_info.num_py_batches = 1
    calibration_data = perf_sweep.get_calibration_inputs(sw_config)

    lgf = perf_sweep.read_graph(sw_config)
    opu_edges = _get_opu_layer_graph(lgf,
                                     calibration_data,
                                     opu_only=opu_only,
                                     max_nodes=max_nodes)
    logging.info("opu edges: {0}".format(opu_edges))

    pruned_graph = lgf.prune_graph(output_edges=opu_edges)
    runner = graph_runner.GraphRunner(pruned_graph, hw_specs, sw_config, sim_params)

    original_shard_outs = _get_outputs(perf_sweep, runner, sw_config)

    # Now get modified
    orig_path = os.path.join(output_dir, "pruned_lgf.pb")
    modified_path = os.path.join(output_dir, "modified_lgf.pb")
    lgf.write_lgf_pb(lgf.as_lgf_pb(), orig_path)

    full_graph_pipeline.main(orig_path,
                             graph_types_pb2.LGFProtobuf,
                             modified_path,
                             graph_types_pb2.LGFProtobuf,
                             calibration_data,
                             hw_specs,
                             sw_config,
                             sim_params)

    modified_lgf = lgf_graph.LightGraph.lgf_pb_to_graph(
        lgf_graph.LightGraph.read_lgf_pb(modified_path))
    modified_pruned_lgf = modified_lgf.prune_graph(output_edges=opu_edges)
    runner = graph_runner.GraphRunner(modified_pruned_lgf,
                                      hw_specs,
                                      sw_config,
                                      sim_params)
    modified_shard_outs = _get_outputs(perf_sweep, runner, sw_config)

    assert (len(original_shard_outs) == len(modified_shard_outs))

    # flatten
    orig_data = {}
    mod_data = {}
    logging.info("num shards: {0}".format(len(original_shard_outs)))
    for i in range(len(original_shard_outs)):
        logging.info("num batches: {0}".format(len(original_shard_outs[i].batches)))
        collect_data_dict(orig_data, original_shard_outs[i])
        collect_data_dict(mod_data, modified_shard_outs[i])

    errors = {}
    for name in orig_data.keys():
        errors[name] = py_graph_test_util.GraphTestCase.relative_error(
            np.concatenate(orig_data[name],
                           axis=1),
            np.concatenate(mod_data[name],
                           axis=1))

    logging.info(errors)

    if fix_bns:
        corrections = {}
        for e in opu_edges:
            if e.name.endswith("FusedBatchNorm"):
                # get per-channel means
                orig = np.concatenate(orig_data[e.name], axis=1)
                mod = np.concatenate(mod_data[e.name], axis=1)
                orig = np.reshape(orig, [-1, orig.shape[-1]])
                mod = np.reshape(mod, [-1, mod.shape[-1]])

                node = modified_lgf.get_node_by_name(e.name)
                corrections[node.inputs[1].name] = np.mean(mod,
                                                           axis=0) - np.mean(orig,
                                                                             axis=0)

        bn_offset_filter = node_filters.name_in_filter("bn_offset")
        offset_transformer = BNOffsetTransform(corrections,
                                               hw_specs,
                                               sim_params,
                                               sw_config)
        node_map = {bn_offset_filter: offset_transformer}

        # Transform graph
        graph_transform = apply_node_map.ApplyNodeMap(hw_specs, sw_config, node_map)
        new_graph = graph_transform.process_transforms(modified_pruned_lgf)
        lgf_graph.LightGraph.write_lgf_pb(new_graph.as_lgf_pb(),
                                          os.path.join(output_dir,
                                                       "fixed_bns.pb"))

    errors_pb = performance_data_pb2.RelativeErrorData()
    errors_pb.errors_dict.update(errors)
    return [errors[e.name] for e in opu_edges], [e.name for e in opu_edges], errors_pb


def _plot_errors(errors, names, plot_path):
    from matplotlib import pyplot as plt
    plt.plot(errors)
    plt.ylabel("Relative Error")
    plt.xticks(np.arange(len(names)), names, rotation="vertical")
    plt.subplots_adjust(bottom=0.55)
    plt.savefig(plot_path)


def main(workload_name, output_dir, hw_cfg, **kwargs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    errors, names, errors_pb = get_layer_errors(workload_name, output_dir, hw_cfg,
                                                **kwargs)
    with open(os.path.join(output_dir, "relative_error_data.pb"), "wb") as f:
        f.write(errors_pb.SerializeToString())

    _plot_errors(errors,
                 names,
                 os.path.join(output_dir,
                              "{0}_errors.png".format(workload_name)))

    return errors, names, errors_pb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        type=str,
        help="workload name, must be a string in performance_sweep_map.py")
    parser.add_argument("--output_dir", type=str, help="directory to save plot")
    parser.add_argument("--opu_only", action="store_true", default=False)
    parser.add_argument("--fix_bns", action="store_true", default=False)
    parser.add_argument("--max_nodes", type=int, default=32)
    parser.add_argument("--config_name",
                        type=str,
                        default=hardware_configs_pb2.HardwareConfig.Name(
                            hardware_configs_pb2.DELTA))

    args = parser.parse_args()

    py_test_util.PythonTestProgram.set_root_logger(logging_level=logging.INFO,
                                                   logging_format="%(message)s")
    main(args.workload,
         args.output_dir,
         hardware_configs_pb2.HardwareConfig.Value(args.config_name),
         opu_only=args.opu_only,
         max_nodes=args.max_nodes,
         fix_bns=args.fix_bns)
