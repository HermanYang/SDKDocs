import os

import numpy as np

from lt_sdk.verification.workloads import streamed_standard_data_base


class StandardDataBase(streamed_standard_data_base.StreamedStandardDataBase):
    """Standard Data Base with a single file for test data"""

    TEST_DATA_FILE = "test_data.npy"
    TEST_LABELS_FILE = "test_labels.npy"

    def num_test_shards(self):
        return 1

    def get_test_inputs(self, sw_config, i):
        return self._get_inputs([self.TEST_DATA_FILE], sw_config)

    def get_test_labels(self, i):
        return np.load(os.path.join(self._data_dir, self.TEST_LABELS_FILE))
