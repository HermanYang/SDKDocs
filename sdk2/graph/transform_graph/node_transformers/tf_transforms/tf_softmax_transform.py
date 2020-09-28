from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    softmax_transform
from sdk2.graph.transform_graph.node_transformers.tf_transforms import tf_base_transform


class TFSavedModelSoftmaxTransform(tf_base_transform.TFSavedModelBaseTransform,
                                   softmax_transform.SoftmaxTransform):

    def transform(self, softmax_node, light_graph):
        """
        Converts original node to a supported softmax in standard format
        """
        self.check_original_node(softmax_node, graph_type=self.GRAPH_TYPE)

        # Get axis
        axis = len(softmax_node.inputs[0].shape.d) - 1

        return self.do_generic_transform(softmax_node.name, softmax_node.inputs[0],
                                         softmax_node.outputs[0],
                                         softmax_node.control_inputs, axis)
