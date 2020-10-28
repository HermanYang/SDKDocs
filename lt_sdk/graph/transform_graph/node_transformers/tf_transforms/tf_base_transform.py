import tensorflow as tf

from lt_sdk.graph.transform_graph.node_transformers import node_transform
from lt_sdk.proto import graph_types_pb2


class TFSavedModelBaseTransform(node_transform.NodeTransform):
    """
    Base class for TFSavedModel node transforms
    """

    GRAPH_TYPE = graph_types_pb2.TFSavedModel

    @staticmethod
    def _get_tf_attr(lnf):
        node_def = tf.NodeDef()
        node_def.ParseFromString(lnf.original.serialized_node)
        return node_def.attr

    @staticmethod
    def _get_string(tf_attr, k):
        return tf.compat.as_str_any(tf_attr[k].s)

    @staticmethod
    def _get_array(tf_attr, k="value"):
        return tf.make_ndarray(tf_attr[k].tensor)

    def _get_array_from_input_indx(self, node, light_graph, input_indx):
        const_node = light_graph.get_node_by_name(node.inputs[input_indx].name)
        return self._get_array(self._get_tf_attr(const_node))
