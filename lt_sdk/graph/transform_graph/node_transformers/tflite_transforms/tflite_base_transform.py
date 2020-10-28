from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    base_transform,
)
from lt_sdk.proto import graph_types_pb2


class TFLiteBaseTransform(base_transform.BaseTransform):
    """
    Base class for TFSavedModel node transforms
    """

    GRAPH_TYPE = graph_types_pb2.TFLiteSavedModel

    def _is_quantized(self):
        return True
