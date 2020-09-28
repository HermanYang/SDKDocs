from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    concat_transform
from sdk2.graph.transform_graph.node_transformers.tf_transforms import tf_base_transform


class TFSavedModelConcatV2Transform(tf_base_transform.TFSavedModelBaseTransform,
                                    concat_transform.ConcatTransform):

    def transform(self, concat_node, light_graph):
        """
        Converts original node to a supported concat in standard format
        """
        self.check_original_node(concat_node, graph_type=self.GRAPH_TYPE)

        # Get axis
        axis = self._get_array_from_input_indx(concat_node, light_graph,
                                               len(concat_node.inputs) - 1)

        return self.do_generic_transform(concat_node.name, concat_node.inputs[:-1],
                                         concat_node.outputs[0],
                                         concat_node.control_inputs, axis)

    def can_transform(self, node, light_graph):
        tf_attr = self._get_tf_attr(node)
        # Get num of input edges
        num = tf_attr["N"].i

        return num == len(node.inputs) - 1 and super().can_transform(node, light_graph)
