from sdk2.graph.transform_graph.graph_transformers import graph_transform
from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    base_transform
from sdk2.proto import lgf_pb2


class AddUpdateVariablesNodes(graph_transform.GraphTransform):

    UPDATE_VARIABLES_NAME = "update_variables"

    def __init__(self, update_info):
        """
        Params:
            update_info: a lgf_pb2.UpdateScalesNode.UpdateInfo() protobuf
        """
        self._update_info = update_info

    def get_transforms(self, light_graph):
        # Create a node that updates variables
        update_variables_node = lgf_pb2.LNF()
        update_variables_node.name = self.UPDATE_VARIABLES_NAME
        update_variables_node.supported = True
        update_variables_node.update_variables.SetInParent()
        update_variables_node.update_variables.update_info.CopyFrom(self._update_info)

        update_variables_node.control_inputs.extend(
            light_graph.output_node_names() +
            [e.name for e in light_graph.output_edges()])

        return [
            base_transform.BaseTransform.create_transform_result(
                to_add=[update_variables_node],
                to_output_swap=[(light_graph.output_node_names(),
                                 [update_variables_node.name])])
        ]
