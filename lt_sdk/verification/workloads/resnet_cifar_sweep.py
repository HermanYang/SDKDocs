from lt_sdk.proto import graph_types_pb2
from lt_sdk.verification.workloads import model_quality, standard_data_base


class ResNetCIFARSweep(standard_data_base.StandardDataBase,
                       model_quality.AccuracyWorkload):

    def input_names(self):
        return ["input_example_tensor"]

    def compilation_batch_size(self):
        return 5

    def py_batch_size(self):
        return 640

    def base_data_dir(self):
        return "cifar10_data"

    def graph_type(self):
        return graph_types_pb2.TFSavedModel

    def prediction_edge(self):
        return "softmax_tensor"


class ResNet8CIFARSweep(ResNetCIFARSweep):

    def graph_dir(self):
        return "resnet8"


class ResNet50CIFARSweep(ResNetCIFARSweep):

    def graph_dir(self):
        return "resnet50"


class ResNet50SparseCIFARSweep(ResNetCIFARSweep):

    def graph_dir(self):
        return "resnet50_sparse_rows_10"
