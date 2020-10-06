from lt_sdk.graph.run_graph.run_external_graph import lgf_graph_runner, tf_graph_runner
from lt_sdk.proto import graph_types_pb2

EXTERNAL_GRAPH_RUNNER_MAP = {
    graph_types_pb2.LGFProtobuf: lgf_graph_runner.LGFProtobufGraphRunner,
    graph_types_pb2.TFSavedModel: tf_graph_runner.TFSavedModelGraphRunner
}
