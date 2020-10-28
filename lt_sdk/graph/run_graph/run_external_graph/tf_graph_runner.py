import tensorflow as tf

from lt_sdk.graph.import_graph import tf_saved_model_base_importer
from lt_sdk.graph.run_graph.run_external_graph import external_graph_runner
from lt_sdk.graph.transform_graph import utils
from lt_sdk.graph.transform_graph.graph_transformers import add_update_variables_nodes
from lt_sdk.inference import tf_ops
from lt_sdk.proto import inference_pb2, lgf_pb2


class TFSavedModelGraphRunner(external_graph_runner.ExternalGraphRunner):
    """
    Class for running a TFSavedModel graph
    """

    @staticmethod
    def _tf_tensor_from_edge(edge_info, graph):
        tf_tensor_name = "{0}:{1}".format(edge_info.name, edge_info.port)
        tf_tensor = graph.get_tensor_by_name(tf_tensor_name)
        return tf_tensor

    @staticmethod
    def inference_input_to_feed_dict(inf_input, graph):
        feed_dict = {}
        for named_tensor in inf_input.inputs:
            tf_tensor = TFSavedModelGraphRunner._tf_tensor_from_edge(
                named_tensor.edge_info,
                graph)
            array = utils.tensor_pb_to_array(
                named_tensor.data,
                utils.dtype_pb_to_np_dtype(named_tensor.data.dtype))
            feed_dict[tf_tensor] = array

        return feed_dict

    @staticmethod
    def _get_lgf_pb_from_tf_node_def(node_def):
        lgf_pb = lgf_pb2.LGF()
        lgf_pb.ParseFromString(node_def.attr["serialized_subgraph"].s)
        return lgf_pb

    @staticmethod
    def init_update_variables_node(sess, output_node_names, feed_dict):
        # TODO: Debug why we need this here? This is a pretty janky solution
        # TF doesn't seem to be executing control inputs from our
        # custom update variables node in the correct order.
        # Here we run all its control inputs once
        for node_name in output_node_names:
            op = sess.graph.get_operation_by_name(node_name)
            if op.type == "LGFSubgraph":
                lgf_pb = TFSavedModelGraphRunner._get_lgf_pb_from_tf_node_def(
                    op.node_def)

                if any(out_name == add_update_variables_nodes.AddUpdateVariablesNodes.
                       UPDATE_VARIABLES_NAME for out_name in lgf_pb.output_node_names):
                    # There should only be one of these nodes in the graph
                    sess.run(op.control_inputs, feed_dict=feed_dict)
                    return

    def _get_stats_pb_fetches(self, graph):
        stats_pb_fetches = []
        for n in graph.as_graph_def().node:
            if n.op == "LGFSubgraph":
                port = len(n.attr["DstT"].list.type) - 1
                tensor_name = "{}:{}".format(n.name, port)
                stats_pb_fetches.append(graph.get_tensor_by_name(tensor_name))

        return stats_pb_fetches

    def run(self, inputs):
        # Load custom ops
        tf_ops.load_ops()

        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            # Load the saved model
            meta_graph_def = tf.saved_model.loader.load(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                self._graph_path)

            # Inputs to feed_dict
            feed_dict = self.inference_input_to_feed_dict(inputs, graph)

            # Get the fetches for the output tensors
            output_fetches = []
            output_tensor_names = (
                tf_saved_model_base_importer.ImportTFSavedModelBase.
                get_output_tensor_names_from_meta_graph_def(meta_graph_def))
            for tensor_name in output_tensor_names:
                output_fetches.append(graph.get_tensor_by_name(tensor_name))

            # Get the fetches for the output nodes
            output_node_names = (
                tf_saved_model_base_importer.ImportTFSavedModelBase.
                get_output_node_names_from_meta_graph_def(meta_graph_def))
            for node_name in output_node_names:
                output_fetches.append(graph.get_operation_by_name(node_name))

            # Add fetches for serialized protobufs
            fetches = output_fetches + self._get_stats_pb_fetches(graph)

            # Extra initialization for custom ops if necessary
            self.init_update_variables_node(sess, output_node_names, feed_dict)

            # Run the graph
            np_outputs = sess.run(fetches, feed_dict=feed_dict)

        # Convert np_outputs to out_pb
        out_pb = inference_pb2.InferenceOutput()
        for i in range(len(output_tensor_names)):
            array = np_outputs[i]
            named_tensor = out_pb.results.add()

            name, port, _ = tf_saved_model_base_importer.ImportTFSavedModelBase.\
                get_node_name_and_output_index(output_tensor_names[i])
            edge_info = named_tensor.edge_info
            edge_info.name = name
            edge_info.port = port
            edge_info.dtype.CopyFrom(
                tf_saved_model_base_importer.ImportTFSavedModelBase.
                tf_dtype_to_lgf_dtype(fetches[i].dtype))
            edge_info.shape.d.extend(array.shape)

            named_tensor.data.CopyFrom(utils.array_to_tensor_pb(array, edge_info.dtype))

        # Get the stats
        stats_list = []
        for i in range(len(output_fetches), len(np_outputs)):
            stats = inference_pb2.ExecutionStats()
            stats.ParseFromString(np_outputs[i].tostring())
            stats_list.append(stats)

        # Get execution stats
        out_pb.stats.CopyFrom(self.get_combined_stats(stats_list))

        return out_pb
