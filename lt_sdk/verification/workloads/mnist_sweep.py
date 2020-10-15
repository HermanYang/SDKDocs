import numpy as np

from lt_sdk.proto import graph_types_pb2
from lt_sdk.verification.workloads import model_quality, standard_data_base


class MnistSweep(standard_data_base.StandardDataBase, model_quality.AccuracyWorkload):

    def base_data_dir(self):
        return "mnist_data"

    def graph_type(self):
        return graph_types_pb2.TFSavedModel

    def compilation_batch_size(self):
        return 10

    def py_batch_size(self):
        return 250

    def input_names(self):
        return ["input_example_tensor"]

    def prediction_edge(self):
        return "ArgMax"

    def output_is_argmax(self):
        return True


class MnistSmallDNNSweep(MnistSweep):

    def graph_dir(self):
        return "small_dnn"

    def logits_tensor_name(self):
        return "logits:0"


class MnistCNNSweep(MnistSweep):

    def graph_dir(self):
        return "cnn"

    def logits_tensor_name(self):
        return "BiasAdd:0"


class MnistSmallDNNTF2Sweep(MnistSweep):

    def graph_type(self):
        return graph_types_pb2.TFGraphDef

    def graph_dir(self):
        return "small_dnn_tf_2/graph.pb"

    def input_names(self):
        return ["Placeholder"]

    def logits_tensor_name(self):
        return "sequential/dense_1/BiasAdd:0"

    def prediction_edge(self):
        return "Identity"

    def output_is_argmax(self):
        return False

    def preprocess(self, array):
        return np.reshape(array, (-1, 28, 28, 1))
