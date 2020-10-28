import os
import shutil

from lt_sdk.graph import lgf_graph
from lt_sdk.graph.run_graph.run_external_graph import external_graph_runner
from lt_sdk.inference import lt_inference
from lt_sdk.proto.configs import utils


class LGFProtobufGraphRunner(external_graph_runner.ExternalGraphRunner):
    """
    Class for running a LGFProtobuf graph
    """

    def _get_dir(self, fname):
        return os.path.join(self._sw_config.debug_info.debug_dir, fname)

    def run(self, inputs):
        if self._sw_config.debug_info.debug_dir:
            utils.write_hw_specs(self._hw_spec, self._get_dir("hw_specs.pb"))
            utils.write_sim_params(self._sim_params, self._get_dir("sim_params.pb"))
            utils.write_sw_config(self._sw_config, self._get_dir("sw_config.pb"))
            shutil.copy2(self._graph_path, self._get_dir("graph.pb"))
            with open(self._get_dir("inputs.pb"), "wb") as f:
                f.write(inputs.SerializeToString())

        # Inference on lgf_pb
        lgf_pb = lgf_graph.LightGraph.read_lgf_pb(self._graph_path)
        out_pb = lt_inference.run_inference(lgf_pb,
                                            inputs,
                                            self._hw_spec,
                                            self._sw_config,
                                            self._sim_params,
                                            graph_coll=self._graph_coll)

        return out_pb
