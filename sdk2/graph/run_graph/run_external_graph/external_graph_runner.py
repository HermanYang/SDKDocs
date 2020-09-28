from sdk2.graph.graph_collections import graph_collection
from sdk2.proto import inference_pb2


class ExternalGraphRunner(object):
    """
    Interface for running exported LightGraph objects that have been exported
    and saved to a graph_path using a ExportGraph object

    Running an exported graph requires code to perform inference with an
    external library (like TensorFlow or PyTorch) or a wrapper to call our
    own custom inference scripts.
    """

    def __init__(self, graph_path, hw_spec, sw_config, sim_params, graph_coll=None):
        """
        Params:
            graph_path: path to a file or directory storing an exported graph
            hw_specs: a hw_specs_pb2.HardwareSpecs() protobuf
            sw_config: a sw_config_pb2.SoftwareConfig() protobuf
            sim_params: a sim_params_pb2.SimulationParams() protobuf
            graph_coll: a graph_collection.GraphCollection() object paired
                with the given graph_path
        """
        self._graph_path = graph_path
        self._hw_spec = hw_spec
        self._sw_config = sw_config
        self._sim_params = sim_params
        self._graph_coll = graph_coll or graph_collection.NullGraphCollection()

    @staticmethod
    def get_combined_stats(stats_list):
        combined = inference_pb2.ExecutionStats()
        for stats in stats_list:
            for inst in stats.instructions:
                combined.instructions.add().CopyFrom(inst)
                combined.instructions[-1].start_clk += combined.total_clocks

            combined.total_clocks += stats.total_clocks

        return combined

    def run(self, inputs):
        """
        Params:
            inputs: a inference_pb2.InferenceInput() protobuf

        Returns:
            out_pb: a inference_pb2.InferenceOutput() protobuf object
        """
        raise NotImplementedError()
