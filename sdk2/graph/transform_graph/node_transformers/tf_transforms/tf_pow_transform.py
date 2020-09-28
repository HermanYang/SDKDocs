import logging

from sdk2.graph.transform_graph.node_transformers.generic_transforms import sv_transform
from sdk2.graph.transform_graph.node_transformers.tf_transforms import tf_base_transform


class TFSavedModelPowTransform(tf_base_transform.TFSavedModelBaseTransform,
                               sv_transform.SVPowTransform):

    def transform(self, pow_node, light_graph):
        """
        Converts original node to a supported transpose in standard format
        """
        self.check_original_node(pow_node, graph_type=self.GRAPH_TYPE)

        power_value = self._get_array_from_input_indx(pow_node, light_graph, 1)

        if (len(power_value.shape) != 0):
            logging.warning("Node transformation only supports scalar power.")
            return self.create_transform_result()

        return self.do_generic_transform(pow_node.name, pow_node.inputs[0],
                                         pow_node.outputs[0], pow_node.control_inputs,
                                         power_value)

    def can_transform(self, node, light_graph):
        return (len(node.inputs) == 2 and self.check_single_output(node, light_graph))
