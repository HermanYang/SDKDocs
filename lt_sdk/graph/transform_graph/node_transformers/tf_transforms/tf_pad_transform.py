from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    pad_transform,
)
from lt_sdk.graph.transform_graph.node_transformers.tf_transforms import (
    tf_base_transform,
)


class TFSavedModelPadTransform(tf_base_transform.TFSavedModelBaseTransform,
                               pad_transform.PadTransform):

    def transform(self, pad_node, light_graph):
        """
        Converts original node to a supported pad in standard format
        """
        self.check_original_node(pad_node, graph_type=self.GRAPH_TYPE)

        # Get padding
        padding = self._get_array_from_input_indx(pad_node, light_graph, 1)

        return self.do_generic_transform(pad_node.name,
                                         pad_node.inputs[0],
                                         pad_node.outputs[0],
                                         pad_node.control_inputs,
                                         padding)


class TFSavedModelUnsupportedPadTransform(TFSavedModelPadTransform):

    # Create an unsupported but transformed pad node. They should get
    # removed by the remove_pad_nodes stage of the base transforms.
    def is_supported(self):
        return False
