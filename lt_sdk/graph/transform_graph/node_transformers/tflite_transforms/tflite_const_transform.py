import numpy as np

from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    const_transform,
)
from lt_sdk.graph.transform_graph.node_transformers.tflite_transforms import (
    tflite_base_transform,
)
from lt_sdk.graph.transform_graph.utils import dtype_pb_to_np_dtype


class TFLiteConstTransform(tflite_base_transform.TFLiteBaseTransform,
                           const_transform.ConstTransform):

    def transform(self, const_node, light_graph):
        # Get the original sub_node
        sub_node = const_node.original

        # Assertions
        self.check_original_node(const_node)

        # Extract the tensor of the const node
        tensor = sub_node.attr["value"].t
        shape = [d for d in tensor.shape.d]
        return self.do_generic_transform(
            const_node.name,
            np.ndarray(shape,
                       dtype=dtype_pb_to_np_dtype(tensor.dtype),
                       buffer=tensor.tensor_content).astype(np.float32))
