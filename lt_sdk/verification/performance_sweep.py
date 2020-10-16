import logging
import os
import shutil

import numpy as np

from lt_sdk.common import py_file_utils
from lt_sdk.graph import full_graph_pipeline, lgf_graph
from lt_sdk.graph.graph_collections import graph_collection
from lt_sdk.graph.import_graph import graph_importer_map
from lt_sdk.graph.run_graph import graph_runner, histogram_graph_runner
from lt_sdk.graph.transform_graph.graph_transformers import convert_to_debug_mode
from lt_sdk.proto import graph_types_pb2, performance_data_pb2, sim_params_pb2
from lt_sdk.visuals import sim_result_to_trace

SWEEP_NAME = "performance_sweep_data.pb"


class PerformanceSweep(object):
    """Interface for performance sweeps"""

    def __init__(self,
                 output_dir,
                 default_data_dir=None,
                 graph_path=None,
                 fine_tuning_fn=None):
        """
        Params:
            output_dir: directory to store the sweep data
            default_data_dir: default top level directory where data and graphs are
                stored
            graph_path: graph path to use when default_data_dir is not provided
            fine_tuning_fn: optional function to use for fine tuning
        """
        self._output_dir = output_dir
        self._graph_type = self.graph_type()
        self._data_dir = ""
        self._default_data_dir = ""
        self._base_data_dir = ""
        self._fine_tuning_fn = fine_tuning_fn
        if graph_path:
            self._graph_path = graph_path
        elif default_data_dir:
            self._default_data_dir = default_data_dir
            if not self.base_data_dir().startswith("/"):
                self._base_data_dir = os.path.join(self._default_data_dir,
                                                   self.base_data_dir())
            else:
                self._base_data_dir = self.base_data_dir()
            self._data_dir = os.path.join(self._base_data_dir, self.data_dir())
            self._graph_path = os.path.join(self.base_graph_dir(), self.graph_dir())
        else:
            raise ValueError("Must specify either default_data_dir or graph_path")

    def get_calibration_inputs(self, sw_config):
        """
        Returns:
            calibration_inputs: an inference_pb2.BatchedInferenceInput() object
                corresponding padded and batched calibration data
        """
        raise NotImplementedError()

    def num_test_shards(self):
        """Returns the number of test shards."""
        raise NotImplementedError()

    def get_test_inputs(self, sw_config, shard_indx):
        """
        Returns:
            test_inputs: an inference_pb2.BatchedInferenceInput() object
                corresponding padded and batched test data for the given
                shard_indx
        """
        raise NotImplementedError()

    def get_test_labels(self, shard_indx):
        """Return labels for the given shard_indx."""
        raise NotImplementedError()

    def num_fine_tuning_shards(self):
        """Return the number of fine tuning shards."""
        return NotImplementedError()

    def get_fine_tuning_inputs(self, sw_config, shard_indx):
        """
        Returns:
            fine_tuning_inputs: an inference_pb2.BatchesInferenceInput() object
                corresponding batched fine_tuning data for the given shard_indx.
                Padding is not allowed.
        """
        raise NotImplementedError()

    def get_fine_tuning_labels(self, sw_config, shard_indx):
        """
        Return fine tuning labels for the given shard_indx.

        Padding is not allowed.
        """
        raise NotImplementedError()

    def logits_tensor_name(self):
        """Return name of logits tensor for fine tuning."""
        raise NotImplementedError()

    def update_quality_metrics(self, performance_data, test_outputs, labels):
        """
        Params:
            performance_data: a performance_pb2.PerformanceData() protobuf
            test_outputs: a inference_pb2.BatchedInferenceOutput() object, where
                test_outputs.batches[i] corresponds to the outputs from
        test_inputs.batches[i]
            labels: The label information. Could be a single numpy array or
                something more complicated if necessary.

        Mutates the given performance_data protobuf to have the most up to date
        quality metrics
        """
        raise NotImplementedError()

    def base_data_dir(self):
        """Returns the base data directory, if something other than
        performance_sweep_map.DATA_DIR
        """
        return NotImplementedError()

    def base_graph_dir(self):
        return self._base_data_dir

    def data_dir(self):
        """Return the data dir relative to the base data dir."""
        return "datasets"

    def full_data_dir(self):
        """Return the absolute path for data dir."""
        return self._data_dir

    def graph_dir(self):
        """Return the trained graph dir relative to what data_dir() returns."""
        raise NotImplementedError()

    def graph_type(self):
        """Return the graph_types_pb2.GraphType of the stored graph."""
        raise NotImplementedError()

    def compilation_batch_size(self):
        """Return the compilation batch size to use."""
        raise NotImplementedError()

    def py_batch_size(self):
        """Return the python batch size to use"""
        raise NotImplementedError()

    def ignore_nodes_filter(self):
        """
        Override to return a node_filters.NodeFilter() object to filter out nodes to
        ignore during graph processing
        """
        return None

    def init_new_config(self):
        """Override to re-initialize class variables for each new config in the sweep"""
        pass

    def end_of_config(self):
        """Override to clean up class variables at the end of a config in the sweep"""
        pass

    def _get_execution_stats(self, performance_data, test_inputs, test_outputs):
        performance_data.execution_stats.CopyFrom(test_outputs.batches[0].stats)

    def get_importer(self, sw_config):
        input_edges = (full_graph_pipeline.extract_edge_from_data(
            self.get_test_inputs(sw_config,
                                 0)))
        return graph_importer_map.GRAPH_IMPORTER_MAP[self._graph_type](
            self._graph_path,
            sw_config,
            input_edges=input_edges)

    def read_graph(self, sw_config):
        importer = self.get_importer(sw_config)
        return importer.as_light_graph()

    def _copy_proto(self, proto):
        copy = proto.__class__()
        copy.CopyFrom(proto)
        return copy

    def _save_debug_info(self,
                         performance_data,
                         cal_hist_pb_map=None,
                         plot_title_map=None):
        sw_config = performance_data.config.sw_config
        hw_specs = performance_data.config.hw_specs
        sim_params = performance_data.config.sim_params

        if sw_config.debug_info.debug_dir:
            sim_result_to_trace.instruction_trace(self.get_trace_path(sw_config),
                                                  performance_data.execution_stats,
                                                  hw_specs,
                                                  sim_params)

            lgf_graph.LightGraph.write_lgf_pb(
                performance_data.graph,
                os.path.join(sw_config.debug_info.debug_dir,
                             "lgf.pb"))

            if sw_config.sweep_info.collect_memory_layout:
                with open(os.path.join(sw_config.debug_info.debug_dir,
                                       "mem_layout.pb"),
                          "wb") as f:
                    f.write(performance_data.simulation_metrics.memory_layout.
                            SerializeToString())

            if sw_config.sweep_info.convert_graph_to_debug_mode:
                assert (cal_hist_pb_map is not None)

                # Make hist directories
                hist_dir = os.path.join(sw_config.debug_info.debug_dir, "histograms")
                if os.path.exists(hist_dir):
                    shutil.rmtree(hist_dir)
                os.makedirs(hist_dir)

                # Save the protobufs
                protobuf_dir = os.path.join(hist_dir, "protobufs")
                os.mkdir(protobuf_dir)
                for key, cal_hist_pb in cal_hist_pb_map.items():
                    hist_path = os.path.join(protobuf_dir,
                                             plot_title_map.get(key,
                                                                str(key)) + ".pb")
                    with open(hist_path, "wb") as f:
                        f.write(cal_hist_pb.SerializeToString())

    def _run_streamed_test_data(self, runner, performance_data):
        # Run data through the graph
        for shard_indx in range(self.num_test_shards()):
            logging.info("-Running inference on test data shard {}".format(shard_indx))
            test_inputs = self.get_test_inputs(performance_data.config.sw_config,
                                               shard_indx)
            test_outputs = runner.run(test_inputs)
            # Update quality metrics each shard
            self.update_quality_metrics(performance_data,
                                        test_outputs,
                                        self.get_test_labels(shard_indx))

        # Just use last shard for execution stats
        self._get_execution_stats(performance_data, test_inputs, test_outputs)

    def _init_graph_coll(self, light_graph, graph_coll, performance_data):
        # Unpack performance data
        sw_config = performance_data.config.sw_config
        hw_specs = performance_data.config.hw_specs

        # Simulation metrics
        sim_metrics_coll = graph_coll.simulation_metrics_collection()
        sim_metrics_coll.set_collect_bit_activity(
            sw_config.sweep_info.collect_bit_activity)
        sim_metrics_coll.set_collect_memory_layout(
            sw_config.sweep_info.collect_memory_layout)
        sim_metrics_coll.initialize_simulation_metrics(hw_specs)

        # Default values to return
        run_graph = light_graph
        runner_cls = graph_runner.GraphRunner
        debug_kwargs = {}

        # Special cases
        if sw_config.sweep_info.convert_graph_to_debug_mode:
            hist_coll = graph_coll.histogram_collection()
            transform = convert_to_debug_mode.ConvertToDebugMode(sw_config, hist_coll)

            run_graph = transform.process_transforms(light_graph)
            runner_cls = histogram_graph_runner.HistogramGraphRunner
            debug_kwargs = {"plot_title_map": transform.get_key_map()}

        return run_graph, runner_cls, debug_kwargs

    def _get_extra_debug_kwargs(self, debug_kwargs, graph_coll, performance_data):
        sw_config = performance_data.config.sw_config

        # Special cases
        if sw_config.sweep_info.convert_graph_to_debug_mode:
            cal_hist_pb_map = {
                k: graph_coll.histogram_collection().get_histogram(k)
                for k in debug_kwargs["plot_title_map"]
            }
            debug_kwargs.update({"cal_hist_pb_map": cal_hist_pb_map})

    def _run_single_config_helper(self, performance_data):
        """Run the config and update performance_data"""
        sw_config = performance_data.config.sw_config
        hw_specs = performance_data.config.hw_specs
        sim_params = performance_data.config.sim_params

        # Use defaults from perf_sweep if necessary
        if sw_config.sweep_info.py_batch_size == 0:
            sw_config.sweep_info.py_batch_size = self.py_batch_size()

        sim_params.compiled_batch_size = self.compilation_batch_size()
        if sw_config.sweep_info.num_py_batches > 0:
            sim_params.compiled_batch_size = min(
                sw_config.sweep_info.num_py_batches * sw_config.sweep_info.py_batch_size,
                sim_params.compiled_batch_size)

        # Graph transformations
        if performance_data.config.do_transform:
            transform_hw_specs = self._copy_proto(hw_specs)
            transform_sw_config = self._copy_proto(sw_config)
            transform_sim_params = self._copy_proto(sim_params)

            transform_sw_config.debug_info.debug_dir = ""
            transform_sim_params.arch_params.arch_type = \
                sim_params_pb2.ArchitectureParams.VIRTUAL

            # Full graph pipeline
            calibration_data = self.get_calibration_inputs(transform_sw_config)
            tmp_dir = py_file_utils.mkdtemp()
            lgf_pb_path = os.path.join(tmp_dir, "modified_lgf.pb")

            full_graph_pipeline.main(self._graph_path,
                                     self._graph_type,
                                     lgf_pb_path,
                                     graph_types_pb2.LGFProtobuf,
                                     calibration_data,
                                     transform_hw_specs,
                                     transform_sw_config,
                                     transform_sim_params)

            # Read light graph
            light_graph = lgf_graph.LightGraph.lgf_pb_to_graph(
                lgf_graph.LightGraph.read_lgf_pb(lgf_pb_path))

            # Cleanup
            shutil.rmtree(tmp_dir)
        else:
            light_graph = self.read_graph(performance_data.config.sw_config)

        # Fine tuning
        if (performance_data.config.do_fine_tuning
                and sw_config.sweep_info.num_fine_tuning_epochs > 0):
            if self._fine_tuning_fn is None:
                raise ValueError("Must provide fine tuning function")

            num_fine_tuning_shards = self.num_fine_tuning_shards()
            tot_num_shards = int(sw_config.sweep_info.num_fine_tuning_epochs *
                                 num_fine_tuning_shards)
            # Get an ordered list of shards to be used for fine tuning
            shard_list = []
            while len(shard_list) < tot_num_shards:
                shard_list.extend(np.random.permutation(range(num_fine_tuning_shards)))
            shard_list = shard_list[:tot_num_shards]
            for i, shard_indx in enumerate(shard_list):
                fine_tuning_data = self.get_fine_tuning_inputs(
                    performance_data.config.sw_config,
                    shard_indx)
                fine_tuning_labels = self.get_fine_tuning_labels(
                    performance_data.config.sw_config,
                    shard_indx)
                light_graph = self._fine_tuning_fn(light_graph,
                                                   fine_tuning_data,
                                                   fine_tuning_labels,
                                                   performance_data.config.hw_specs,
                                                   performance_data.config.sw_config,
                                                   performance_data.config.sim_params,
                                                   self.logits_tensor_name())

        # Create debug_dir if necessary
        debug_dir = sw_config.debug_info.debug_dir
        if debug_dir:
            if os.path.exists(debug_dir):
                shutil.rmtree(debug_dir)
            os.makedirs(debug_dir)

        with graph_collection.GraphCollection() as graph_coll:
            # Initialize graph for running test data
            run_graph, runner_cls, debug_kwargs = self._init_graph_coll(
                light_graph, graph_coll, performance_data)

            # Run test data
            runner = runner_cls(light_graph, hw_specs, sw_config, sim_params, graph_coll)
            self._run_streamed_test_data(runner, performance_data)

            # Get extra information after running
            self._get_extra_debug_kwargs(debug_kwargs, graph_coll, performance_data)

            # Save simulation metrics
            performance_data.simulation_metrics.CopyFrom(
                graph_coll.simulation_metrics_collection().get_simulation_metrics())

        # Save graph and debug info
        performance_data.graph.CopyFrom(light_graph.as_lgf_pb())
        self._save_debug_info(performance_data, **debug_kwargs)

    def run_single_config(self, config, indx):
        """
        Params:
            config: a performance_data_pb2.ConfigInfo() protobuf
            indx: unique index for the config

        Returns:
            performance_data: a performance_data_pb2.PerformanceData() protobuf
        """
        # Initialize some things
        self.init_new_config()

        performance_data = performance_data_pb2.PerformanceData()
        performance_data.config.CopyFrom(config)
        performance_data_path = os.path.join(self._output_dir,
                                             "performance_data_{}.pb".format(indx))

        # Do not re-run a config if it is already on disk
        if os.path.exists(performance_data_path):
            logging.warning(
                ("Found performance data on disk, skipping configuration {0}. " +
                 "To re-run this configuration, remove or rename the " +
                 "following file: {1}").format(indx,
                                               performance_data_path))
            with open(performance_data_path, "rb") as f:
                performance_data.ParseFromString(f.read())
        else:
            self._run_single_config_helper(performance_data)
            with open(performance_data_path, "wb") as f:
                f.write(performance_data.SerializeToString())

        # Clean up and return
        self.end_of_config()
        return performance_data

    def get_trace_path(self, sw_config):
        return os.path.join(sw_config.debug_info.debug_dir, "batch0.trace")

    def save_performance_sweep_data(self, performance_sweep_data):
        performance_sweep_data_path = os.path.join(self._output_dir, SWEEP_NAME)
        with open(performance_sweep_data_path, "wb") as f:
            f.write(performance_sweep_data.SerializeToString())

    def run_configs(self, configs):
        """
        Params:
            configs: a list of performance_data_pb2.ConfigInfo() protobufs

        Returns:
            performance_sweep_data: a performance_data_pb2.PerformanceSweepData()
                protobuf where performance_data.data[i] corresponds to configs[i]
        """
        # Sweep all configs
        performance_sweep_data = performance_data_pb2.PerformanceSweepData()
        for indx, config in enumerate(configs):
            logging.info("------- Running config {0}: {1} ---------".format(
                indx,
                config.description))
            performance_data = self.run_single_config(config, indx)
            performance_sweep_data.data.add().CopyFrom(performance_data)

        return performance_sweep_data
