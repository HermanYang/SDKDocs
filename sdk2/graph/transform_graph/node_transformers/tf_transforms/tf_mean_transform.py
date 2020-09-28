from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    mean_transform
from sdk2.graph.transform_graph.node_transformers.tf_transforms import tf_base_transform


class TFSavedModelMeanTransform(tf_base_transform.TFSavedModelBaseTransform,
                                mean_transform.MeanTransform):

    def transform(self, mean_node, light_graph):
        """
        Converts original node to a supported mean in standard format
        """
        self.check_original_node(mean_node, graph_type=self.GRAPH_TYPE)

        # Get axes
        axes = self._get_array_from_input_indx(mean_node, light_graph, 1).flatten()
        tf_attr = self._get_tf_attr(mean_node)
        keep_dims = tf_attr["keep_dims"].b

        return self.do_generic_transform(mean_node.name, mean_node.inputs[0],
                                         mean_node.outputs[0], mean_node.control_inputs,
                                         axes, keep_dims)
