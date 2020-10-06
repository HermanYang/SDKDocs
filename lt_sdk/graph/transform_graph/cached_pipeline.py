import logging
import os

from lt_sdk.graph import lgf_graph
from lt_sdk.graph.transform_graph import utils


class StageArgs(object):

    def __init__(self,
                 calibration_data,
                 original_graph,
                 hw_specs,
                 sw_config,
                 sim_params):
        self.calibration_data = calibration_data
        self.original_graph = original_graph
        self.hw_specs = hw_specs
        self.sw_config = sw_config
        self.sim_params = sim_params


class PipelineStage(object):

    def __init__(self, name, stage_args):
        self.name = name
        self._hw_specs = stage_args.hw_specs
        self._sw_config = stage_args.sw_config
        self._sim_params = stage_args.sim_params

    def execute(self, light_graph):
        """Return a new light_graph."""
        raise NotImplementedError()

    def _cache_fname(self):
        return os.path.join(self._sw_config.cache_dir,
                            "{0}.pb".format(self.name.replace(" ",
                                                              "")))

    def load(self):
        """Return light_graph if loaded, None if not."""
        fname = self._cache_fname()
        if os.path.exists(fname):
            lgf_pb = lgf_graph.LightGraph.read_lgf_pb(fname)
            return lgf_graph.LightGraph.lgf_pb_to_graph(lgf_pb)
        return None

    def write_stage(self, light_graph):
        fname = self._cache_fname()
        lgf_pb = light_graph.as_lgf_pb()
        lgf_graph.LightGraph.write_lgf_pb(lgf_pb, fname)

    def transform(self, light_graph):
        if self._sw_config.cache_dir:
            cached = self.load()
            if cached:
                logging.info("-Loaded from cached file!")
                return cached

        logging.info("-Executing.")
        new_graph = self.execute(light_graph)
        if self._sw_config.cache_dir:
            self.write_stage(new_graph)
        return new_graph


class FnStage(PipelineStage):

    def __init__(self, fn, name, stage_args):
        super().__init__(name, stage_args)
        self._fn = fn

    def execute(self, light_graph):
        return self._fn(light_graph)


class CachedPipeline(object):
    """Saves and reuses the result of stages."""

    def __init__(self, stages):
        """Stages are subclasses of PipelineStage."""
        self._stages = stages

    def transform(self, light_graph):
        for s in self._stages:
            utils.log_message("Running stage: {0}".format(s.name))
            light_graph = s.transform(light_graph)
        return light_graph
