import os

import tensorflow as tf

from lt_sdk.common import py_file_utils
from lt_sdk.graph.export_graph import tf_graph_exporter
from lt_sdk.graph.import_graph import (
    tf_saved_model_base_importer,
    tf_saved_model_importer,
)
from lt_sdk.graph.transform_graph.graph_transformers import graph_transform
from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    base_transform,
)
from lt_sdk.proto import graph_types_pb2, lgf_pb2, ops_pb2, transform_result_pb2


class AddCastNodes(graph_transform.GraphTransform):

    def __init__(self, sw_config):
        self._sw_config = sw_config

    def _get_new_node_name(self, name):
        if name in self._new_node_names:
            i = 1
            while name + "_{}".format(i) in self._new_node_names:
                i += 1

            name += "_{}".format(i)

        self._new_node_names.add(name)
        return name

    @staticmethod
    def _get_original_tf_node(lnf_node, sw_config, op=ops_pb2.CAST):
        g = tf.Graph()
        with g.as_default():
            with tf.Session(graph=g) as sess:
                dtype_tup = (lnf_node.inputs[0].dtype.t, lnf_node.inputs[0].dtype.p)
                inp = tf.placeholder(
                    tf_saved_model_base_importer.ImportTFSavedModelBase.
                    REV_DATA_TYPE_MAP[dtype_tup],
                    shape=[None if d < 0 else d for d in lnf_node.inputs[0].shape.d],
                    name=lnf_node.inputs[0].name)
                dtype_tup = (lnf_node.outputs[0].dtype.t, lnf_node.outputs[0].dtype.p)
                tf_cast = tf.cast(inp,
                                  tf_saved_model_base_importer.ImportTFSavedModelBase.
                                  REV_DATA_TYPE_MAP[dtype_tup],
                                  name=lnf_node.name)

                tmp_dir = py_file_utils.mkdtemp()
                saved_model_dir = os.path.join(tmp_dir, "saved_model")
                tf_graph_exporter.ExportTFSavedModel.save_model(
                    saved_model_dir,
                    sess,
                    [inp],
                    [tf_cast])
        importer = tf_saved_model_importer.ImportTFSavedModel(saved_model_dir, sw_config)
        light_graph = importer.as_light_graph()

        tf_node = light_graph.get_node_by_name(lnf_node.name)
        return tf_node.original

    @staticmethod
    def process_cast_node(cast_node, input_node, output_node, sw_config):
        if input_node.supported and output_node.supported:
            # Add a supported cast node between two supported nodes
            cast_node.supported = True
            cast_node.cast.SetInParent()
        else:
            # Otherwise use external library for cast node
            cast_node.supported = False
            if input_node.HasField(lgf_pb2.LNF.original.DESCRIPTOR.name):
                graph_type = input_node.original.t
            elif output_node.HasField(lgf_pb2.LNF.original.DESCRIPTOR.name):
                graph_type = output_node.original.t
            else:
                raise ValueError("Either input or output should have original node")

            # TODO: this will only work for TF graphs.
            if graph_type in {
                    graph_types_pb2.TFSavedModel,
                    graph_types_pb2.TFTrainingSavedModel,
                    graph_types_pb2.TFTrainingCheckpoint
            }:
                cast_node.original.CopyFrom(
                    AddCastNodes._get_original_tf_node(cast_node,
                                                       sw_config))
            else:
                raise ValueError(
                    "Could not transform constant node with graph type: {}".format(
                        graph_type))

    def _add_cast_node(self, node, input_edge, input_node, matching_output_edge):
        """
        Adds a cast node between the edges, so the graph
        will have input_node --{matching_output_edge}--> cast --{input_edge}--> node
        """
        # Create a cast node
        cast_node = lgf_pb2.LNF()
        cast_node.name = self._get_new_node_name(matching_output_edge.name + "_cast")

        # Input and output info
        cast_node.inputs.add().CopyFrom(matching_output_edge)
        cast_node.outputs.add().CopyFrom(input_edge)
        cast_node.outputs[0].name = cast_node.name
        cast_node.outputs[0].port = 0

        self.process_cast_node(cast_node, input_node, node, self._sw_config)

        # Create transform result
        return base_transform.BaseTransform.create_transform_result(
            to_add=[cast_node],
            to_reroute=[(transform_result_pb2.ToReroute.edge_reroute.DESCRIPTOR.name,
                         [node.name],
                         input_edge,
                         cast_node.outputs[0])])

    def get_transforms(self, light_graph):
        self._new_node_names = set()
        transforms = []

        for n in light_graph.nodes():
            for input_edge in n.inputs:
                if light_graph.has_node(input_edge.name):
                    input_node = light_graph.get_node_by_name(input_edge.name)
                    matching_output_edge = input_node.outputs[input_edge.port]

                    # Add a cast node between edges with inconsistent dtypes
                    if (input_edge.dtype.SerializeToString() !=
                            matching_output_edge.dtype.SerializeToString()):
                        transforms.append(
                            self._add_cast_node(n,
                                                input_edge,
                                                input_node,
                                                matching_output_edge))

        return transforms
