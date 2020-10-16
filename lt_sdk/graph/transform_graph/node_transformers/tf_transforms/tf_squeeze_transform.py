from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    reshape_transform,
)
from lt_sdk.graph.transform_graph.node_transformers.tf_transforms import (
    tf_base_transform,
)


class TFSavedModelSqueezeTransform(tf_base_transform.TFSavedModelBaseTransform,
                                   reshape_transform.ReshapeTransform):

    def transform(self, squeeze_node, light_graph):
        """
        Converts original node to a supported reshape in standard format
        """
        self.check_original_node(squeeze_node, graph_type=self.GRAPH_TYPE)

        # Check only one input for now
        assert (len(squeeze_node.inputs) == 1)

        return self.do_generic_transform(squeeze_node.name,
                                         squeeze_node.inputs[0],
                                         squeeze_node.outputs[0],
                                         squeeze_node.control_inputs)
