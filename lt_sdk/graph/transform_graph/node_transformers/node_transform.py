class NodeTransform(object):
    """Interface for a node transform object"""

    def __init__(self, hw_specs, sw_config, sim_params):
        """
        Params:
            hw_specs: a hw_spec_pb2.HardwareSpecs() protobuf
            sw_config: a sw_config_pb2.SoftwareConfig() protobuf
            sim_params: a sim_params_pb2.SimulationParams() protobuf
        """
        self._hw_specs = hw_specs
        self._sw_config = sw_config
        self._sim_params = sim_params

    def can_transform(self, node, light_graph):
        """Returns true if the transform should run."""
        return True

    def transform(self, node, light_graph):
        """Return new nodes to replace given node in graph.

        params:
            node: A LightNode object.
            light_graph: A LightGraph object.
        returns:
            A TransformResult object
        """
        raise NotImplementedError()

    def _common_args(self):
        return (self._hw_specs, self._sw_config, self._sim_params)
