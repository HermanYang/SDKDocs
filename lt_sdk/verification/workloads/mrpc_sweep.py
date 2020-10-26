import os

import numpy as np

from lt_sdk.proto import dtypes_pb2, graph_types_pb2, node_filters
from lt_sdk.verification.workloads import model_quality, standard_data_base


class MRPCSweep(standard_data_base.StandardDataBase, model_quality.AccuracyWorkload):
    """Base class for BERT-based workload performance sweep."""

    MAX_SEQ_LENGTH = 128

    CALIB_DATA_FILE_FORMAT = "calibration_data_{}.npy"
    TEST_DATA_FILE_FORMAT = "test_data_{}.npy"

    def base_data_dir(self):
        return "mrpc_data"

    def input_names(self):
        return ["input_ids", "segment_ids", "input_mask"]

    def compilation_batch_size(self):
        return 1

    def py_batch_size(self):
        return 100

    def graph_type(self):
        return graph_types_pb2.TFSavedModel

    def get_calibration_inputs(self, sw_config):
        return self._get_inputs(
            [self.CALIB_DATA_FILE_FORMAT.format(name) for name in self.input_names()],
            sw_config,
            allow_padding=False,
            dtype=dtypes_pb2.DT_INT)

    def get_test_inputs(self, sw_config, _):
        return self._get_inputs(
            [self.TEST_DATA_FILE_FORMAT.format(name) for name in self.input_names()],
            sw_config,
            dtype=dtypes_pb2.DT_INT)

    def get_test_labels(self, _):
        return np.load(
            os.path.join(self._data_dir,
                         self.TEST_DATA_FILE_FORMAT.format("label_ids")))


class BERTMRPCSweep(MRPCSweep):

    def graph_dir(self):
        return "bert"

    def logits_tensor_name(self):
        return "loss/BiasAdd:0"

    def prediction_edge(self):
        return "loss/Softmax"

    def ignore_nodes_filter(self):
        ignore_nodes = [
            node_filters.name_is_filter("bert/embeddings/Reshape"),
            node_filters.name_is_filter("bert/embeddings/Reshape_2")
        ]
        return node_filters.or_filter(*ignore_nodes)
