from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    fill_transform
from sdk2.graph.transform_graph.node_transformers.tf_transforms import tf_base_transform


class TFSavedModelFillTransform(tf_base_transform.TFSavedModelBaseTransform,
                                fill_transform.FillTransform):

    def transform(self, fill_node, light_graph):
        """
        Converts original node to a supported const in standard format
        """
        self.check_original_node(fill_node, graph_type=self.GRAPH_TYPE)

        return self.do_generic_transform(
            fill_node.name, fill_node.outputs[0],
            self._get_array_from_input_indx(fill_node, light_graph, 1))
