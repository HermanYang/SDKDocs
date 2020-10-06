from lt_sdk.graph.import_graph import (
    lgf_graph_importer,
    onnx_saved_model_importer,
    tf_graph_def_importer,
    tf_saved_model_importer,
    tf_training_checkpoint_importer,
    tf_training_saved_model_importer,
)
from lt_sdk.proto import graph_types_pb2

GRAPH_IMPORTER_MAP = {
    graph_types_pb2.LGFProtobuf:
        lgf_graph_importer.ImportLGFProtobuf,
    graph_types_pb2.TFSavedModel:
        tf_saved_model_importer.ImportTFSavedModel,
    graph_types_pb2.TFGraphDef:
        tf_graph_def_importer.ImportTFGraphDef,
    graph_types_pb2.TFTrainingSavedModel:
        tf_training_saved_model_importer.ImportTFTrainingSavedModel,
    graph_types_pb2.TFTrainingCheckpoint:
        tf_training_checkpoint_importer.ImportTFCheckpoint,
    graph_types_pb2.ONNXModel:
        onnx_saved_model_importer.ImportONNXModel
}
