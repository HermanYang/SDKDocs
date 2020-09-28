import glob
import os

import numpy as np

from sdk2.data import batch, named_tensor
from sdk2.proto import dtypes_pb2
from sdk2.verification import performance_sweep


class StreamedStandardDataBase(performance_sweep.PerformanceSweep):
    """Standard Data Base that streams test files from disk"""

    # TODO: support streamed calibration data? would require a lot more plumbing
    # Also, change to protobuf? or keep as numpy?
    CALIBRATION_DATA_FILE = "calibration_data.npy"
    CALIBRATION_LABELS_FILE = "calibration_labels.npy"
    TEST_DATA_FILE_FORMAT = "test_data_shard_{}.npy"
    TEST_LABELS_FILE_FORMAT = "test_labels_shard_{}.npy"
    FINE_TUNING_DATA_FILE_FORMAT = "fine_tuning_data_shard_{}.npy"
    FINE_TUNING_LABELS_FILE_FORMAT = "fine_tuning_labels_shard_{}.npy"

    def input_names(self):
        """Returns names of input edges"""
        raise NotImplementedError()

    def preprocess(self, array: np.ndarray):
        """preprocesses input arrayarray

        Args:
            array (np.ndarray): single input array

        Returns:
            [np.ndarray]: preprocessed input array
        """
        return array

    def data_subset(self, data, sw_config):
        """Return the subset of the data to use."""
        if sw_config.sweep_info.num_py_batches > 0:
            return data[:sw_config.sweep_info.num_py_batches *
                        sw_config.sweep_info.py_batch_size]

        return data

    def get_data_array(self, fname):
        if not fname.startswith(self._data_dir):
            fname = os.path.join(self._data_dir, fname)
        return np.load(fname)

    def _get_inputs(self,
                    filenames,
                    sw_config,
                    allow_padding=True,
                    dtype=dtypes_pb2.DT_FLOAT):
        assert len(filenames) == len(self.input_names())

        names = self.input_names()
        tensors = [self.get_data_array(filename) for filename in filenames]
        named_tensors = named_tensor.NamedTensorSet(names, tensors, dtype=dtype)

        named_tensors.apply_all(self.preprocess)

        def get_subset(arr):
            return self.data_subset(arr, sw_config)

        named_tensors.apply_all(get_subset)

        batched_inputs = batch.batch_inputs(
            named_tensors,
            batch_size=sw_config.sweep_info.py_batch_size,
            allow_padding=allow_padding)

        return batched_inputs

    def get_calibration_inputs(self, sw_config):
        return self._get_inputs([self.CALIBRATION_DATA_FILE],
                                sw_config,
                                allow_padding=False)

    def _sorted_paths(self, file_format):
        return sorted(glob.glob(os.path.join(self._data_dir, file_format.format("*"))))

    def num_test_shards(self):
        return len(self._sorted_paths(self.TEST_DATA_FILE_FORMAT))

    def get_test_inputs(self, sw_config, i):
        return self._get_inputs([self._sorted_paths(self.TEST_DATA_FILE_FORMAT)[i]],
                                sw_config)

    def get_test_labels(self, i):
        return np.load(self._sorted_paths(self.TEST_LABELS_FILE_FORMAT)[i])

    def num_fine_tuning_shards(self):
        return len(self._sorted_paths(self.FINE_TUNING_DATA_FILE_FORMAT))

    # Do not allow padding for fine tuning unless we know that padded data are treated
    # correctly (i.e., discarded) during fine tuning
    def get_fine_tuning_inputs(self, sw_config, i):
        return self._get_inputs(
            [self._sorted_paths(self.FINE_TUNING_DATA_FILE_FORMAT)[i]],
            sw_config,
            allow_padding=False)

    def get_fine_tuning_labels(self, sw_config, i):
        fine_tuning_labels = np.load(
            self._sorted_paths(self.FINE_TUNING_LABELS_FILE_FORMAT)[i])
        return batch.pad_and_batch_tensor(fine_tuning_labels,
                                          sw_config.sweep_info.py_batch_size,
                                          allow_padding=False)
