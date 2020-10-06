import tensorflow as tf

from lt_sdk.graph.import_graph import (
    tf_saved_model_importer,
    tf_training_checkpoint_importer,
)


class ImportTFGraphDef(tf_saved_model_importer.ImportTFSavedModel):

    def __init__(self, graph_path, *args, graph_def=None, graph=None, **kwargs):
        self._graph_def = tf.GraphDef()
        if graph is not None:
            self._graph_def.CopyFrom(graph.as_graph_def())
        elif graph_def is not None:
            self._graph_def.CopyFrom(graph_def)
        else:
            with open(graph_path, "rb") as f:
                self._graph_def.ParseFromString(f.read())

        super().__init__(graph_path, *args, **kwargs)

    def _get_output_tensor_and_node_names(self):
        self._output_tensor_names = []
        self._output_node_names = []

        with tf.Session(graph=tf.Graph()) as sess:
            tf.import_graph_def(self._graph_def, name="")

            all_inputs = set()
            all_control_inputs = set()
            for op in sess.graph.get_operations():
                for inp in op.inputs:
                    all_inputs.add(inp.name)
                for ctrl in op.control_inputs:
                    all_control_inputs.add(ctrl.name)

            for op in sess.graph.get_operations():
                for out in op.outputs:
                    if out.name not in all_inputs:
                        # Assume a tensor that is not an input to any other
                        # op in the graph is an output tensor of the whole graph
                        self._output_tensor_names.append(out.name)
                if (len(op.outputs) == 0
                        and (len(op.inputs) > 0 or len(op.control_inputs) > 0)):
                    if op.name not in all_control_inputs:
                        # Assume an op that has no outputs and is not a control
                        # input for another op in the graph is an output node
                        # of the whole graph
                        self._output_node_names.append(op.name)

    def load_graph_session(self, read_graph_fn):
        self._input_tensor_names = (tf_training_checkpoint_importer.ImportTFCheckpoint.
                                    _get_placeholder_tensors_from_subgraph(
                                        self._graph_def))
        self._get_output_tensor_and_node_names()
        const_graph_def = self._fold_constants(self._graph_def)
        self._load_const_graph_def(const_graph_def, read_graph_fn)
