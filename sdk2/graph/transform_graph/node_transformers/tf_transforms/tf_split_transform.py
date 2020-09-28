from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    split_transform
from sdk2.graph.transform_graph.node_transformers.tf_transforms import tf_base_transform


class TFSavedModelSplitTransform(tf_base_transform.TFSavedModelBaseTransform,
                                 split_transform.SplitTransform):

    def transform(self, split_node, light_graph):
        """
        Converts original node to a supported split in standard format
        """
        self.check_original_node(split_node, graph_type=self.GRAPH_TYPE)

        # Get axis
        if len(split_node.inputs) == 2:  # Equal split
            axis = self._get_array_from_input_indx(split_node, light_graph, 0)
            input_indx = 1
        else:  # Unequal split
            axis = self._get_array_from_input_indx(split_node, light_graph, 2)
            input_indx = 0

        return self.do_generic_transform(split_node.name, split_node.inputs[input_indx],
                                         split_node.outputs, split_node.control_inputs,
                                         axis)

    def can_transform(self, node, light_graph):
        tf_attr = self._get_tf_attr(node)
        # Get num of output edges
        num = tf_attr["num_split"].i

        return len(node.inputs) in (2, 3) and num == len(
            node.outputs) and super().can_transform(node, light_graph)
