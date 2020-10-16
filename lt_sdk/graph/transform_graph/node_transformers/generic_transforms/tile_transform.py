from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    electronic_op_transform,
)
from lt_sdk.proto import lgf_pb2


class TileTransform(electronic_op_transform.ElectronicOpTransform):

    def create_supported_nodes(self, tile_name, input_edge, output_edge, control_inputs):
        """
        Creates a supported tile node in standard format

        Params:
            tile_name: name of original node
            input_edge: edge of the input for the original node
            output_edge: edge of the output for the original node
            control_inputs: a list of node names for the control inputs
        """
        return [
            self.create_simple_node(tile_name,
                                    lgf_pb2.LNF.tile.DESCRIPTOR.name,
                                    [input_edge],
                                    [output_edge],
                                    control_inputs)
        ]

    def can_transform(self, tile_node, light_graph):
        return len(tile_node.inputs) == 2 and self.check_single_output(
            tile_node,
            light_graph)

    def transform(self, tile_node, light_graph):
        """
        Generic TileTransform
        """
        self.check_original_node(tile_node)
        return self.do_generic_transform(tile_node.name,
                                         tile_node.inputs[0],
                                         tile_node.outputs[0],
                                         tile_node.control_inputs)
