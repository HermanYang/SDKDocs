import ctypes
import os

from lt_sdk.common import context_manager_utils
from lt_sdk.graph.graph_collections import graph_collection
from lt_sdk.proto import inference_pb2


class PyInferenceRunner(object):
    """
    Wrapper around a cpp class
    Instances of this class should ALWAYS be created with a context manager
    """

    lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "lib_py_wrapper.so"))

    lib.CreatePyInferenceRunner.restype = ctypes.c_void_p
    lib.DeletePyInferenceRunner.argtypes = (ctypes.c_void_p,)

    lib.Run.argtypes = (ctypes.c_void_p,
                        ctypes.c_char_p,
                        ctypes.c_uint,
                        ctypes.c_char_p,
                        ctypes.c_uint,
                        ctypes.c_char_p,
                        ctypes.c_uint,
                        ctypes.c_char_p,
                        ctypes.c_uint,
                        ctypes.c_char_p,
                        ctypes.c_uint,
                        ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_uint))
    lib.Run.restype = ctypes.POINTER(ctypes.c_char)

    lib.ClearPyInferenceRunnerProtoString.argtypes = (ctypes.c_void_p,)

    @context_manager_utils.__init__
    def __init__(self):
        self._obj = self.lib.CreatePyInferenceRunner()

    @context_manager_utils.__enter__
    def __enter__(self):
        return self

    @context_manager_utils.__exit__
    def __exit__(self, exc_type, exc_value, traceback):
        self.lib.DeletePyInferenceRunner(self._obj)

    @context_manager_utils.force_in_context
    def run(self, lgf_pb, inputs, hw_spec, sw_config, sim_params, graph_coll):
        spec_data = hw_spec.SerializeToString()
        sw_config_data = sw_config.SerializeToString()
        params_data = sim_params.SerializeToString()
        lgf_data = lgf_pb.SerializeToString()
        input_data = inputs.SerializeToString()
        inf_out_size = ctypes.c_uint()

        inf_out_ptr = self.lib.Run(self._obj,
                                   spec_data,
                                   len(spec_data),
                                   sw_config_data,
                                   len(sw_config_data),
                                   params_data,
                                   len(params_data),
                                   lgf_data,
                                   len(lgf_data),
                                   input_data,
                                   len(input_data),
                                   graph_coll.pointer(),
                                   ctypes.byref(inf_out_size))

        inf_out = inference_pb2.InferenceOutput()
        inf_out.ParseFromString(inf_out_ptr[:inf_out_size.value])

        self.lib.ClearPyInferenceRunnerProtoString(self._obj)

        return inf_out


def run_inference(lgf_pb, inputs, hw_spec, sw_config, sim_params, graph_coll=None):
    graph_coll = graph_coll or graph_collection.NullGraphCollection()
    with PyInferenceRunner() as py_inf_runner:
        inf_out = py_inf_runner.run(lgf_pb,
                                    inputs,
                                    hw_spec,
                                    sw_config,
                                    sim_params,
                                    graph_coll)

    return inf_out
