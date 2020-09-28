from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    stack_transform
from sdk2.graph.transform_graph.node_transformers.tf_transforms import tf_base_transform


class TFSavedModelStackTransform(tf_base_transform.TFSavedModelBaseTransform,
                                 stack_transform.StackTransform):

    def transform(self, stack_node, light_graph):
        """
        Converts original node to a supported stack in standard format
        """
        self.check_original_node(stack_node, graph_type=self.GRAPH_TYPE)

        tf_attr = self._get_tf_attr(stack_node)
        # Get axis
        axis = tf_attr["axis"].i

        return self.do_generic_transform(stack_node.name, stack_node.inputs,
                                         stack_node.outputs[0],
                                         stack_node.control_inputs, axis)

    def can_transform(self, node, light_graph):
        tf_attr = self._get_tf_attr(node)
        # Get num of input edges
        num = tf_attr["N"].i

        return num == len(node.inputs) and super().can_transform(node, light_graph)
