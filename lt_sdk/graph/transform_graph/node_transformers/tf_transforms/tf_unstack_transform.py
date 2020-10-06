from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    unstack_transform,
)
from lt_sdk.graph.transform_graph.node_transformers.tf_transforms import (
    tf_base_transform,
)


class TFSavedModelUnstackTransform(tf_base_transform.TFSavedModelBaseTransform,
                                   unstack_transform.UnstackTransform):

    def transform(self, unstack_node, light_graph):
        """
        Converts original node to a supported unstack in standard format
        """
        self.check_original_node(unstack_node, graph_type=self.GRAPH_TYPE)

        tf_attr = self._get_tf_attr(unstack_node)
        # Get axis
        axis = tf_attr["axis"].i

        return self.do_generic_transform(unstack_node.name,
                                         unstack_node.inputs[0],
                                         unstack_node.outputs,
                                         unstack_node.control_inputs,
                                         axis)

    def can_transform(self, node, light_graph):
        tf_attr = self._get_tf_attr(node)
        # Get num of output edges
        num = tf_attr["num"].i

        return num == len(node.outputs) and super().can_transform(node, light_graph)
