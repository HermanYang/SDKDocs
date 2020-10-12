import logging

from lt_sdk.graph.run_graph import graph_runner
from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    opu_op_transform,
)
from lt_sdk.proto import calibration_pb2, lgf_pb2, node_filters


class HistogramGraphRunner(graph_runner.GraphRunner):
    """
    Graph Runner that can manage histograms/cal data.
    """

    def __init__(self, light_graph, hw_spec, sw_config, sim_params, graph_coll):
        super().__init__(light_graph,
                         hw_spec,
                         sw_config,
                         sim_params,
                         graph_coll=graph_coll)
        self._hist_filter = node_filters.which_oneof_filter(*(
            list(sw_config.node_types.opu_nodes) +
            [lgf_pb2.LNF.collect_hist.DESCRIPTOR.name]))

        # Checks
        assert (not graph_coll.is_null())
        self._hist_coll = graph_coll.histogram_collection()

    def _update_hist_coll(self, hist_keys):
        for index, key in enumerate(hist_keys.keys):
            mode = self._hist_coll.get_histogram_mode(key)

            # Do nothing
            if mode in {calibration_pb2.HM_INVALID, calibration_pb2.HM_PADDING}:
                continue

            # Convert max to populate
            elif mode == calibration_pb2.HM_UPDATE_MAX:
                hist_max_val = self._hist_coll.get_histogram_max_val(key)
                if self._sw_config.ignore_empty_histograms and hist_max_val == 0:
                    # Set max value to arbitrary number
                    self._hist_coll.set_histogram_max_val(key, 1)
                else:
                    # Max value should be non-zero
                    assert (hist_max_val > 0)
                self._hist_coll.update_histogram_mode(key,
                                                      calibration_pb2.HM_POPULATE_HIST)

            # Invalid update
            else:
                raise ValueError(
                    "Cannot update histogram collection mode when current mode is {}".
                    format(mode))

    def convert_to_populate_mode(self):
        for node in self._light_graph.nodes():
            if self._hist_filter.matches(node, self._light_graph):
                if node.HasField(lgf_pb2.LNF.collect_hist.DESCRIPTOR.name):
                    self._update_hist_coll(node.collect_hist.hist_keys)
                elif node.WhichOneof("node") in self._sw_config.node_types.opu_nodes:
                    matmul = opu_op_transform.OPUOpTransform.get_matmul_from_opu_node(
                        node)
                    self._update_hist_coll(matmul.hist_keys_before_adc)
                    self._update_hist_coll(matmul.hist_keys_after_adc)

    def run(self, inputs, output_edges=None):
        """
        Performs super().run() twice. The first time, super().run() updates the
        max values in self._graph_coll.histogram_collection(). The second time,
        super().run() updates the counts in the histograms in
        self._graph_coll.histogram_collection()

        Returns the inference_pb2.BatchedInferenceOutput() produced. These should be the
        same during the first and second time super().run() is called.
        """
        # Collect max
        logging.info("-HistogramGraphRunner collecting max")
        super().run(inputs, output_edges=output_edges)

        self.convert_to_populate_mode()

        # Populate histograms
        logging.info("-HistogramGraphRunner populating histograms")
        return super().run(inputs, output_edges=output_edges)
