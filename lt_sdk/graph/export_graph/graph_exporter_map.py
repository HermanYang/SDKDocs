from lt_sdk.graph.export_graph import lgf_graph_exporter, tf_graph_exporter
from lt_sdk.proto import graph_types_pb2

GRAPH_EXPORTER_MAP = {
    graph_types_pb2.LGFProtobuf: lgf_graph_exporter.ExportLGFProtobuf,
    graph_types_pb2.TFSavedModel: tf_graph_exporter.ExportTFSavedModel,
    graph_types_pb2.TFLiteSavedModel: tf_graph_exporter.ExportTFSavedModel
}
