import tensorflow as tf

from lt_sdk.graph.import_graph import tf_saved_model_base_importer


class ImportTFTrainingSavedModel(tf_saved_model_base_importer.ImportTFSavedModelBase):

    def load_graph_session(self, read_graph_fn):
        with tf.Session(graph=tf.Graph()) as sess:
            self._meta_graph_def = tf.saved_model.loader.load(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                self._graph_path)

            self._input_tensor_names = self.get_input_tensor_names_from_meta_graph_def(
                self._meta_graph_def)
            self._output_tensor_names = self.get_output_tensor_names_from_meta_graph_def(
                self._meta_graph_def)
            self._output_node_names = self.get_output_node_names_from_meta_graph_def(
                self._meta_graph_def)

            read_graph_fn(sess)
