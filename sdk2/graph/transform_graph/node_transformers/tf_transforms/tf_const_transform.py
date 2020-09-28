from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    const_transform
from sdk2.graph.transform_graph.node_transformers.tf_transforms import tf_base_transform


class TFSavedModelConstTransform(tf_base_transform.TFSavedModelBaseTransform,
                                 const_transform.ConstTransform):

    def transform(self, const_node, light_graph):
        """
        Converts original node to a supported const in standard format
        """
        self.check_original_node(const_node, graph_type=self.GRAPH_TYPE)

        return self.do_generic_transform(const_node.name,
                                         self._get_array(self._get_tf_attr(const_node)))
