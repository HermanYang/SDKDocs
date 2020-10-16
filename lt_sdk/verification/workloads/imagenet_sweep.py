import os

import numpy as np

from lt_sdk.proto import graph_types_pb2
from lt_sdk.verification.workloads import (
    model_quality,
    standard_data_base,
    streamed_standard_data_base,
)


class ImageNetSweep(standard_data_base.StandardDataBase, model_quality.AccuracyWorkload):

    def base_data_dir(self):
        return "imagenet_data"

    def compilation_batch_size(self):
        return 5

    def py_batch_size(self):
        return 640

    def graph_type(self):
        return graph_types_pb2.TFSavedModel


class MobileNetSweep(ImageNetSweep):

    def input_names(self):
        return ["input"]

    def graph_dir(self):
        return "mobilenet_v2"

    def prediction_edge(self):
        return "MobilenetV2/Predictions/Reshape_1"

    def preprocess(self, array):
        return array.astype(np.float32) / 128 - 1

    def compilation_batch_size(self):
        return 4

    def py_batch_size(self):
        return 256


class ResNet50Sweep(ImageNetSweep):

    INPUT_NAME = "input_example_tensor"
    GRAPH_DIR = "resnet_AWS_NHWC"
    PREDICTION_EDGE = "softmax_tensor"

    def input_names(self):
        return [self.INPUT_NAME]

    def graph_dir(self):
        return self.GRAPH_DIR

    def prediction_edge(self):
        return self.PREDICTION_EDGE


class FullResnet50Sweep(streamed_standard_data_base.StreamedStandardDataBase,
                        model_quality.AccuracyWorkload):

    def base_data_dir(self):
        # For now this is outside the bazel training_data target because it makes
        # the tar 27GB, which is a little cumbersome.
        return "/data/algo"

    def data_dir(self):
        return "imagenet_test"

    def compilation_batch_size(self):
        return 5

    def py_batch_size(self):
        return 640

    def graph_type(self):
        return graph_types_pb2.TFSavedModel

    def input_names(self):
        return [ResNet50Sweep.INPUT_NAME]

    def base_graph_dir(self):
        return os.path.join(self._default_data_dir, "imagenet_data")

    def graph_dir(self):
        return ResNet50Sweep.GRAPH_DIR

    def prediction_edge(self):
        return ResNet50Sweep.PREDICTION_EDGE


class EfficientNetSweep(ImageNetSweep):

    STDDEV_RGB = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])

    def input_names(self):
        return ["images"]

    def graph_dir(self):
        return "efficientnet_b0"

    def prediction_edge(self):
        return "Softmax"

    def compilation_batch_size(self):
        return 4

    def py_batch_size(self):
        return 256

    def preprocess(self, array):
        return array.astype(np.float32) / EfficientNetSweep.STDDEV_RGB

    # EfficientNet does not output background label (label 0)
    def get_test_labels(self, i):
        return super(EfficientNetSweep, self).get_test_labels(i) - 1

    def get_fine_tuning_labels(self, sw_config, i):
        return super(EfficientNetSweep, self).get_fine_tuning_labels(sw_config, i) - 1
