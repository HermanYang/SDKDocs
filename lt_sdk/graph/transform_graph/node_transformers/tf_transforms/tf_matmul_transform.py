from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    matmul_transform,
)
from lt_sdk.graph.transform_graph.node_transformers.tf_transforms import (
    tf_base_transform,
)


class TFSavedModelMatMulTransform(tf_base_transform.TFSavedModelBaseTransform,
                                  matmul_transform.MatMulTransform):

    def transform(self, matmul_node, light_graph):
        """
        Converts original node to a supported matmul in standard format
        """
        self.check_original_node(matmul_node, graph_type=self.GRAPH_TYPE)
        tf_attr = self._get_tf_attr(matmul_node)

        # Get input and weight nodes
        input_node, weight_node = self.find_input_and_weight_nodes(
            matmul_node, light_graph)
        input_index = self._get_input_index(input_node, matmul_node)

        # Get transpose attributes
        transpose_map = {0: "transpose_a", 1: "transpose_b"}
        transpose_inputs = tf_attr[transpose_map[input_index]].b
        transpose_weights = tf_attr[transpose_map[0 if input_index else 1]].b

        return self.do_generic_transform(matmul_node.name,
                                         matmul_node.inputs[input_index],
                                         weight_node.outputs[0],
                                         matmul_node.outputs[0],
                                         matmul_node.control_inputs,
                                         transpose_inputs=transpose_inputs,
                                         transpose_weights=transpose_weights)
