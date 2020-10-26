from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    electronic_op_transform,
)
from lt_sdk.proto import lgf_pb2


class FillTransform(electronic_op_transform.ElectronicOpTransform):

    def create_supported_nodes(self, fill_name, output_edge, value):
        """
        Creates a supported fill node in standard format

        Params:
            fill_name: name of original node
            output_edge: edge of the output for the original node
            value: a scalar with which to fill the output tensor
        """
        fill_node = self.create_simple_node(fill_name,
                                            lgf_pb2.LNF.fill.DESCRIPTOR.name,
                                            [],
                                            [output_edge],
                                            [])

        fill_node.fill.value = value

        return [fill_node]

    def can_transform(self, fill_node, light_graph):
        return len(fill_node.control_inputs) == 0 and self.check_single_output(
            fill_node,
            light_graph)
