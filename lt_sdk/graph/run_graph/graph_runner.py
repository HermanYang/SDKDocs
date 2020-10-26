import logging
import os
import shutil

from lt_sdk.common import py_file_utils
from lt_sdk.graph.export_graph import graph_exporter, graph_exporter_map
from lt_sdk.graph.graph_collections import graph_collection
from lt_sdk.graph.run_graph.run_external_graph import external_graph_runner_map
from lt_sdk.graph.transform_graph import utils
from lt_sdk.proto import graph_types_pb2, inference_pb2, lgf_pb2


class GraphRunner(object):
    """
    Class that can run any LightGraph
    Relies on ExternalGraphRunner which runs a graph of a unified type
    """

    def __init__(self, light_graph, hw_spec, sw_config, sim_params, graph_coll=None):
        """
        Params:
            light_graph: a LightGraph object
            hw_specs: a hw_specs_pb2.HardwareSpecs() protobuf
            sw_config: a sw_config_pb2.SoftwareConfig() protobuf
            sim_params: a sim_params_pb2.SimulationParams() protobuf
            graph_coll: a graph_collection.GraphCollection() object
        """
        self._light_graph = light_graph
        self._hw_spec = hw_spec
        self._sw_config = sw_config
        self._sim_params = sim_params
        self._graph_coll = graph_coll or graph_collection.NullGraphCollection()
        self._check_consistent_edges(self._light_graph)

    @staticmethod
    def is_fully_supported(light_graph):
        """
        Returns true if light_graph is fully suppported
        """
        return all([n.supported for n in light_graph.nodes()])

    @staticmethod
    def get_graph_type(light_graph):
        """
        Returns a graph_types_pb2.GraphType enum for the graph type
        is needed to run the given light_graph
        """
        if GraphRunner.is_fully_supported(light_graph):
            return graph_types_pb2.LGFProtobuf

        for n in light_graph.nodes():
            if not n.supported and n.HasField(lgf_pb2.LNF.original.DESCRIPTOR.name):
                return n.original.t

    @staticmethod
    def _check_inputs(inputs):
        for inf_inp in inputs.batches:
            for named_tensor in inf_inp.inputs:
                assert (named_tensor.edge_info.shape.SerializeToString() ==
                        named_tensor.data.shape.SerializeToString())
                assert (named_tensor.edge_info.dtype.SerializeToString() ==
                        named_tensor.data.dtype.SerializeToString())
                assert (all(d != -1 for d in named_tensor.edge_info.shape.d))

    @staticmethod
    def _check_consistent_edges(light_graph):
        for n in light_graph.nodes():
            for input_edge in n.inputs:
                if light_graph.has_node(input_edge.name):
                    try:
                        matching_output_edge = \
                            light_graph.get_node_by_name(input_edge.name).\
                            outputs[input_edge.port]
                    except IndexError as e:
                        logging.error("output port {0} not found in node: {1}".format(
                            input_edge.port,
                            light_graph.get_node_by_name(input_edge.name)))
                        logging.error("parent node {0}".format(n))
                        logging.error("in edge: {0}".format(input_edge))
                        logging.fatal(e)
                    if not utils.edges_match(input_edge,
                                             matching_output_edge,
                                             check_consistent=True):
                        raise ValueError(
                            "Found a pair of inconsistent edges.\n" +
                            "Node: {}".format(n) + "\n" +
                            "Input Edge: {}".format(input_edge) + "\n" +
                            "Matching Output Edge: {}".format(matching_output_edge))

    @staticmethod
    def _create_output_map(output_edges, light_graph):
        # Maps edges in output_edges to outputs of light_graph, takes care of scenario
        # where the edges in output_edges get collapsed into a subgraph
        output_map = {}
        for out_edge in output_edges:
            matching_edge = None
            for e_out in light_graph.output_edges():
                # out_edge matches an edge of light_graph
                if utils.edges_match(out_edge, e_out):
                    matching_edge = e_out

                # out_edge matches the output of a subgraph
                elif light_graph.get_node_by_name(e_out.name).HasField(
                        lgf_pb2.LNF.subgraph.DESCRIPTOR.name):
                    subgraph_pb = light_graph.get_node_by_name(e_out.name).subgraph.graph
                    subgraph_edge = subgraph_pb.output_edges[e_out.port]
                    if utils.edges_match(out_edge, subgraph_edge):
                        matching_edge = e_out

            if matching_edge is not None:
                output_map[(out_edge.name,
                            out_edge.port)] = (matching_edge.name,
                                               matching_edge.port)
            else:
                raise RuntimeError("Could not find matching edge")

        return output_map

    @staticmethod
    def _get_aligned_outputs(outputs, output_edges, output_map):
        aligned_outputs = inference_pb2.BatchedInferenceOutput()
        for out_inf in outputs.batches:
            aligned_inf_out = aligned_outputs.batches.add()
            aligned_inf_out.stats.CopyFrom(out_inf.stats)

            # Aligned inference output should have ordering and naming of output_edges
            for out_edge in output_edges:
                name, port = output_map[(out_edge.name, out_edge.port)]
                for named_tensor in out_inf.results:
                    if (named_tensor.edge_info.name == name
                            and named_tensor.edge_info.port == port):
                        corrected_named_tensor = aligned_inf_out.results.add()
                        corrected_named_tensor.CopyFrom(named_tensor)
                        corrected_named_tensor.edge_info.name = out_edge.name
                        corrected_named_tensor.edge_info.port = out_edge.port
                        break

            # Make sure we found everything
            assert (len(out_inf.results) == len(aligned_inf_out.results) ==
                    len(output_edges))

        return aligned_outputs

    def run(self, inputs, output_edges=None):
        """
        Params:
            inputs: a inference_pb2.BatchedInferenceInput() protobuf
            outputs_edges: a list of lgf_pb2.EdgeInfo() protobufs, if None will use
                self._light_graph.outputs()

        Returns:
            outputs: a inference_pb2.BatchedInferenceOutput() protobuf object, such that
                outputs.batches[i].results[j] corresponds to the edge output_edges[j]
                from batch inputs.batches[i]
        """
        # Check inputs and get output_edges
        self._check_inputs(inputs)
        if output_edges is None:
            output_edges = self._light_graph.output_edges()

        # Prune graph
        input_edges = [nt.edge_info for nt in inputs.batches[0].inputs]
        light_graph = self._light_graph.prune_graph(input_edges=input_edges,
                                                    output_edges=output_edges,
                                                    include_inputs=False)

        # Collapse graph if it is not fully supported
        if not (self.is_fully_supported(light_graph)):
            light_graph = graph_exporter.ExportGraph.get_collapsed_light_graph(
                light_graph)

        # Create exporter
        graph_type = self.get_graph_type(light_graph)
        exporter = graph_exporter_map.GRAPH_EXPORTER_MAP[graph_type](
            light_graph,
            self._hw_spec,
            self._sw_config,
            self._sim_params,
            graph_coll=self._graph_coll)

        # Export the graph
        tmp_dir = py_file_utils.mkdtemp()
        graph_path = os.path.join(tmp_dir, "graph_path")
        exporter.export_graph(graph_path)

        # External graph runner
        external_runner = external_graph_runner_map.EXTERNAL_GRAPH_RUNNER_MAP[
            graph_type](graph_path,
                        self._hw_spec,
                        self._sw_config,
                        self._sim_params,
                        graph_coll=self._graph_coll)

        # Run inference
        outputs = inference_pb2.BatchedInferenceOutput()
        for inf_inp in inputs.batches:
            outputs.batches.add().CopyFrom(external_runner.run(inf_inp))

        # Clean up
        shutil.rmtree(tmp_dir)

        # Re-align outputs because outputs of collapsed graph may have different
        # names than output of original graph
        output_map = self._create_output_map(output_edges, light_graph)
        return self._get_aligned_outputs(outputs, output_edges, output_map)

    def run_single_batch(self, inputs, output_edges=None):
        """
        Params:
            inputs: a inference_pb2.InferenceInput() protobuf
            outputs_edges: a list of lgf_pb2.EdgeInfo() protobufs, if None will use
                self._light_graph.outputs()

        Returns:
            outputs: a inference_pb2.InferenceOutput() protobuf object, such that
                outputs.results[j] corresponds to the edge output_edges[j]
        """
        batched_inputs = inference_pb2.BatchedInferenceInput()
        batched_inputs.batches.add().CopyFrom(inputs)
        return self.run(batched_inputs, output_edges=output_edges).batches[0]
