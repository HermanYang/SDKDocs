import logging

import numpy as np

from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    transpose_transform
from sdk2.graph.transform_graph.node_transformers.tf_transforms import tf_base_transform


class TFSavedModelTransposeTransform(tf_base_transform.TFSavedModelBaseTransform,
                                     transpose_transform.TransposeTransform):

    def transform(self, transpose_node, light_graph):
        """
        Converts original node to a supported transpose in standard format
        """
        self.check_original_node(transpose_node, graph_type=self.GRAPH_TYPE)

        # Get axes
        axes = self._get_array_from_input_indx(transpose_node, light_graph, 1).flatten()

        # Update TF defaults
        input_rank = len(transpose_node.inputs[0].shape.d)
        if axes.size == 0:
            axes = np.flip(np.arange(input_rank))
        elif axes.size != input_rank:
            logging.error("Could not transform TF transpose op")
            return self.create_transform_result()

        return self.do_generic_transform(transpose_node.name, transpose_node.inputs[0],
                                         transpose_node.outputs[0],
                                         transpose_node.control_inputs, axes)
