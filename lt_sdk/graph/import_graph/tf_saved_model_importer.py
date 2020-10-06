import tensorflow as tf
from tensorflow.tools import graph_transforms as tf_graph_transforms

from lt_sdk.graph.import_graph import tf_saved_model_base_importer


class ImportTFSavedModel(tf_saved_model_base_importer.ImportTFSavedModelBase):

    def _gen_new_name(self, old_name, i):
        """Modify node name based on the given index i."""
        nn = old_name.split("/")
        nn[-1] = "__" + str(i) + "__"
        return "/".join(nn)

    def _fix_node_names_for_constant_folding(self, const_graph_def):
        """
        Assign fixed names for nodes generated in constant folding.
        The constant folding transformation generates node names with a unique ID that is
        not necessarily the same value across different calls of the transformation. This
        function fixes this problem by assigning unique IDs to the generated nodes based
        on the alphabetic order of their names.
        """

        gen_nodes = [node.name for node in const_graph_def.node if "__cf__" in node.name]
        gen_nodes.sort()
        new_names = {
            name: self._gen_new_name(name,
                                     i) for i,
            name in enumerate(gen_nodes)
        }
        for node in const_graph_def.node:
            if node.name in new_names:
                node.name = new_names[node.name]
            for i, s in enumerate(node.input):
                n = self.get_node_name_and_output_index(s)[0]
                if n in new_names:
                    node.input[i] = node.input[i].replace(n, new_names[n])

    def _fold_constants(self, const_graph_def):
        const_graph_def = tf_graph_transforms.TransformGraph(
            const_graph_def,
            self._input_tensor_names,
            self._output_tensor_names + list(self._output_node_names),
            [
                "fold_constants",
                "remove_nodes(op=Identity, op=CheckNumerics, op=StopGradient)",
                "fold_batch_norms",  # fold_constants must happen before this
                "fold_old_batch_norms"
            ])

        self._fix_node_names_for_constant_folding(const_graph_def)

        return const_graph_def

    def _load_const_graph_def(self, const_graph_def, read_graph_fn):
        # Create a new session with the new constant graph def
        with tf.Session(graph=tf.Graph()) as sess:
            tf.import_graph_def(const_graph_def, name="")

            # Create a meta graph def
            self._meta_graph_def = tf.MetaGraphDef()
            self._meta_graph_def.graph_def.CopyFrom(const_graph_def)

            read_graph_fn(sess)

    def load_graph_session(self, read_graph_fn):
        # Read the graph, convert variables to constants, and fold constants
        with tf.Session(graph=tf.Graph()) as sess:
            meta_graph_def = tf.saved_model.loader.load(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                self._graph_path)
            self._input_tensor_names = self.get_input_tensor_names_from_meta_graph_def(
                meta_graph_def)
            self._output_tensor_names = self.get_output_tensor_names_from_meta_graph_def(
                meta_graph_def)
            self._output_node_names = self.get_output_node_names_from_meta_graph_def(
                meta_graph_def)

            output_node_names = [
                self.get_node_name_and_output_index(t)[0]
                for t in self._output_tensor_names
            ] + list(self._output_node_names)
            const_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess,
                sess.graph_def,
                output_node_names)

            # Replace any node that always evaluates to constant expressions with
            # those constants
            const_graph_def = self._fold_constants(const_graph_def)

        self._load_const_graph_def(const_graph_def, read_graph_fn)
