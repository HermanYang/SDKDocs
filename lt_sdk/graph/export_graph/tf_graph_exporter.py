import os
import struct

import tensorflow as tf
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import constants as tf_constants
from tensorflow.python.saved_model import (
    signature_constants,
    signature_def_utils,
    tag_constants,
)
from tensorflow.python.saved_model import utils as tf_utils

from lt_sdk.graph.export_graph import graph_exporter
from lt_sdk.graph.import_graph import tf_saved_model_base_importer
from lt_sdk.graph.transform_graph import utils
from lt_sdk.inference import tf_ops
from lt_sdk.proto import graph_types_pb2, lgf_pb2, ops_pb2


class ExportTFSavedModel(graph_exporter.ExportGraph):
    """Export graph as TF Saved Model"""

    @staticmethod
    def _get_tf_dtypes(edge_list):
        return [
            tf.dtypes.as_dtype(utils.dtype_pb_to_np_dtype(e.dtype)).as_datatype_enum
            for e in edge_list
        ]

    @staticmethod
    def _get_tensor_names(edge_list):
        return ["{0}:{1}".format(e.name, e.port) for e in edge_list]

    def _subgraph_node_to_node_def(self, node_def, lnf):
        node_def.name = lnf.name
        node_def.op = "LGFSubgraph"
        node_def.attr["SrcT"].list.type.extend(self._get_tf_dtypes(lnf.inputs))
        node_def.attr["DstT"].list.type.extend(self._get_tf_dtypes(lnf.outputs))
        node_def.attr["DstT"].list.type.extend([tf.int8.as_datatype_enum])
        node_def.attr["serialized_subgraph"].s = lnf.subgraph.graph.SerializeToString()
        node_def.attr["serialized_spec"].s = self._hw_specs.SerializeToString()
        node_def.attr["serialized_sw_config"].s = self._sw_config.SerializeToString()
        node_def.attr["serialized_params"].s = self._sim_params.SerializeToString()

        if not self._graph_coll.is_null():
            node_def.attr["graph_coll_addr"].s = struct.pack("L",
                                                             self._graph_coll.address())

        node_def.input.extend(ExportTFSavedModel._get_tensor_names(lnf.inputs))
        node_def.input.extend(["^" + node_name for node_name in lnf.control_inputs])

    @staticmethod
    def _original_node_def(node_def, lnf, variable_values):
        if lnf.original.t != graph_types_pb2.TFSavedModel:
            raise ValueError("original node not a TFSavedModel: {0}".format(lnf.name))

        # Use serialized node def and fix inputs
        node_def.ParseFromString(lnf.original.serialized_node)
        del (node_def.input[:])
        node_def.input.extend(ExportTFSavedModel._get_tensor_names(lnf.inputs))
        for cntrl in lnf.control_inputs:
            node_def.input.append("^" + cntrl)

        # Get values from variables
        if lnf.original.op == ops_pb2.VARIABLE:
            if (tf_saved_model_base_importer.ImportTFSavedModelBase.TENSOR_PB
                    not in lnf.original.attr):
                raise ValueError(
                    "Could not find value associated with variable node: {}".format(lnf))

            tensor_pb = tensor_pb2.TensorProto()
            tensor_pb.ParseFromString(lnf.original.attr[
                tf_saved_model_base_importer.ImportTFSavedModelBase.TENSOR_PB].v)
            variable_values[node_def.name] = tf.make_ndarray(tensor_pb)

    @staticmethod
    def _edge_to_node_def_placeholder(edge):
        node_def = tf.NodeDef()
        node_def.name = edge.name
        node_def.op = "Placeholder"
        node_def.attr["dtype"].type = ExportTFSavedModel._get_tf_dtypes([edge])[0]
        for d in edge.shape.d:
            node_def.attr["shape"].shape.dim.add().size = d

        return node_def

    @staticmethod
    def save_model(saved_model_dir,
                   sess,
                   input_tensors=None,
                   output_tensors=None,
                   output_node_names=None):
        input_tensors = input_tensors or []
        output_tensors = output_tensors or []
        output_node_names = output_node_names or []

        inputs = {t.name: tf_utils.build_tensor_info(t) for t in input_tensors}
        outputs = {t.name: tf_utils.build_tensor_info(t) for t in output_tensors}

        model_signature = signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=signature_constants.PREDICT_METHOD_NAME)

        builder = saved_model_builder.SavedModelBuilder(saved_model_dir)
        builder.add_meta_graph_and_variables(
            sess,
            [tag_constants.SERVING],
            clear_devices=True,
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: model_signature,
            })

        builder.save()

        # Add the output node names to the collection def
        fname = os.path.join(saved_model_dir, tf_constants.SAVED_MODEL_FILENAME_PB)
        saved_model = saved_model_pb2.SavedModel()
        with open(fname, "rb") as f:
            saved_model.ParseFromString(f.read())

        saved_model.meta_graphs[0].collection_def[
            tf_saved_model_base_importer.ImportTFSavedModelBase.
            OUTPUT_NODES_COLLECTION_DEF_KEY].node_list.value.extend(output_node_names)

        with open(fname, "wb") as f:
            f.write(saved_model.SerializeToString())

    @staticmethod
    def _get_tf_tensors_from_graph(graph, edge_list):
        tensor_names = ExportTFSavedModel._get_tensor_names(edge_list)
        return [graph.get_tensor_by_name(n) for n in tensor_names]

    def export_graph(self, saved_model_dir):
        # Collapse supported subgraphs
        light_graph = self.get_collapsed_light_graph(self._light_graph)

        # Check meta graph info
        meta_graph_info = light_graph.meta_graph_info()
        if (tf_saved_model_base_importer.ImportTFSavedModelBase.PARTIAL_META_GRAPH_DEF
                not in meta_graph_info.original_graph_info):
            raise ValueError(
                "Could not find partial meta graph def in meta graph info: {}".format(
                    meta_graph_info))

        # Convert to TF Graph Def
        meta_graph = tf.MetaGraphDef()
        meta_graph.ParseFromString(meta_graph_info.original_graph_info[
            tf_saved_model_base_importer.ImportTFSavedModelBase.PARTIAL_META_GRAPH_DEF].v
                                  )
        graph_def = meta_graph.graph_def
        variable_values = {}
        for lnf in light_graph.nodes():
            node_def = graph_def.node.add()
            if lnf.HasField(lgf_pb2.LNF.subgraph.DESCRIPTOR.name):
                self._subgraph_node_to_node_def(node_def, lnf)
            else:
                self._original_node_def(node_def, lnf, variable_values)

        # Add placeholders for nodes that don't exist in the light_graph
        node_names = set(n.name for n in graph_def.node)
        for e in light_graph.input_edges():
            if not light_graph.has_node(e.name):
                graph_def.node.add().CopyFrom(self._edge_to_node_def_placeholder(e))
                node_names.add(e.name)

        # Load the graph def into a session (load custom ops first)
        tf_ops.load_ops()
        with tf.Session(graph=tf.Graph()) as sess:
            if (meta_graph.saver_def.save_tensor_name != ""):
                # TF only allows importing a meta graph def when the saver def
                # is fully initialized. This should only be the case for graphs
                # with TF variables
                assert (len(variable_values) > 0)
                tf.train.import_meta_graph(meta_graph)
            else:
                assert (len(variable_values) == 0)
                tf.import_graph_def(meta_graph.graph_def, name="")

            # Load variables
            for v in (tf.global_variables() + tf.local_variables() +
                      tf.trainable_variables() + tf.model_variables()):
                node_name, _, _ = tf_saved_model_base_importer.ImportTFSavedModelBase.\
                    get_node_name_and_output_index(v.name)
                if node_name not in variable_values:
                    raise ValueError(
                        "Could not find value for variable {}".format(node_name))
                v.load(variable_values[node_name])

            # Get input and output tensors from the graph
            input_tensors = self._get_tf_tensors_from_graph(sess.graph,
                                                            light_graph.input_edges())
            output_tensors = self._get_tf_tensors_from_graph(sess.graph,
                                                             light_graph.output_edges())

            # Create the saved model
            self.save_model(saved_model_dir,
                            sess,
                            input_tensors=input_tensors,
                            output_tensors=output_tensors,
                            output_node_names=light_graph.output_node_names())
