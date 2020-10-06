from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    electronic_op_transform,
)
from lt_sdk.proto import lgf_pb2


class ConstTransform(electronic_op_transform.ElectronicOpTransform):

    def create_supported_nodes(self,
                               const_name,
                               array,
                               const_type=lgf_pb2.ConstNode.GRAPH_CONST):
        """
        Creates a supported const node in standard format

        Params:
            const_name: name of original node
            array: numpy array
            const_type: type of the constant
        """
        return [self.create_const_node(array, const_name, self._get_dtype(), const_type)]
