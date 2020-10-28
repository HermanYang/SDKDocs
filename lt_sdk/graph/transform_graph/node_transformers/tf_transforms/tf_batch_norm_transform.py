from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    batch_norm_transform,
)
from lt_sdk.graph.transform_graph.node_transformers.tf_transforms import (
    tf_base_transform,
)


class TFSavedModelBatchNormTransform(tf_base_transform.TFSavedModelBaseTransform,
                                     batch_norm_transform.BatchNormTransform):

    def transform(self, bn_node, light_graph):
        """
        Converts original node to a supported batchnorm in standard format
        """
        self.check_original_node(bn_node, graph_type=self.GRAPH_TYPE)

        # Get attributes
        scale = self._get_array_from_input_indx(bn_node, light_graph, 1)
        bias = self._get_array_from_input_indx(bn_node, light_graph, 2)
        mean = self._get_array_from_input_indx(bn_node, light_graph, 3)
        variance = self._get_array_from_input_indx(bn_node, light_graph, 4)
        epsilon = self._get_tf_attr(bn_node)["epsilon"].f

        return self.do_generic_transform(bn_node.name,
                                         bn_node.inputs[0],
                                         bn_node.outputs[0],
                                         bn_node.control_inputs,
                                         mean,
                                         variance,
                                         scale,
                                         bias,
                                         epsilon)


class TFSavedModelDecomposeBatchNormTransform(TFSavedModelBatchNormTransform):

    def decompose(self):
        return True
