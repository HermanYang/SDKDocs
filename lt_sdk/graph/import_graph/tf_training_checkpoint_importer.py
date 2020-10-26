import logging
import os

import tensorflow as tf
from tensorflow.core.protobuf import queue_runner_pb2

from lt_sdk.graph.import_graph import tf_saved_model_base_importer


class ImportTFCheckpoint(tf_saved_model_base_importer.ImportTFSavedModelBase):

    @staticmethod
    def _get_placeholder_tensors_from_subgraph(subgraph):
        return ["{}:0".format(n.name) for n in subgraph.node if n.op == "Placeholder"]

    def load_graph_session(self, read_graph_fn):
        meta_path = None
        for f in os.listdir(self._graph_path):
            if f.endswith(".meta"):
                meta_path = os.path.join(self._graph_path, f)
        if not meta_path:
            raise ValueError("Could not find meta file in: {0}".format(self._graph_path))

        with tf.Session(graph=tf.Graph()) as sess:
            # Load the checkpoint
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess, tf.train.latest_checkpoint(self._graph_path))
            self._meta_graph_def = saver.export_meta_graph()

            # Get the output node names
            self._output_node_names = self.get_output_node_names_from_meta_graph_def(
                self._meta_graph_def)
            if not len(self._output_node_names):
                self._output_node_names.update(self._meta_graph_def.collection_def[
                    tf.GraphKeys.TRAIN_OP].node_list.value)

            assert (len(self._output_node_names) > 0)
            self._output_tensor_names = []

            # Assume inputs are placeholders that the output nodes depend on
            subgraph = tf.graph_util.extract_sub_graph(sess.graph_def,
                                                       list(self._output_node_names))
            self._input_tensor_names = self._get_placeholder_tensors_from_subgraph(
                subgraph)

            # If we did not find any placeholders, try to get placeholders that the
            # queue runner depends on
            if not len(self._input_tensor_names):
                for serialized_proto in self._meta_graph_def.collection_def[
                        tf.GraphKeys.QUEUE_RUNNERS].bytes_list.value:
                    # Deserialize the queue runner proto
                    qr_proto = queue_runner_pb2.QueueRunnerDef()
                    qr_proto.ParseFromString(serialized_proto)

                    # Try to find dequeue ops
                    dequeue_ops = []
                    for node in sess.graph_def.node:
                        for inp_name in node.input:
                            inp_name, _, _ = self.get_node_name_and_output_index(
                                inp_name)
                            if (inp_name == qr_proto.queue_name
                                    and "dequeue" in node.name.lower()):
                                dequeue_ops.append(
                                    sess.graph.get_operation_by_name(node.name))

                    # Input tensor names for the graphs will be output tensors of the
                    # dequeue ops
                    for dequeue_op in dequeue_ops:
                        for out_ten in dequeue_op.outputs:
                            self._input_tensor_names.append(out_ten.name)
                            self._required_nodes.add(dequeue_op.node_def.name)

            # Just print a warning if we could not find any inputs
            if not len(self._input_tensor_names):
                logging.warning("Did not find any input tensors in the graph")

            read_graph_fn(sess)
