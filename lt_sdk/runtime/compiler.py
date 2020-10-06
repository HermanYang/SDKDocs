import ctypes
import os

from lt_sdk.common import context_manager_utils
from lt_sdk.proto import subgraph_binary_pb2


class PyCompiler(object):
    """
    Wrapper around subgraph_context.cc:PyCompiler
    Instances of this class should ALWAYS be created with a context manager
    """

    lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "lib_compiler.so"))

    lib.CreatePyCompiler.restype = ctypes.c_void_p
    lib.DeletePyCompiler.argtypes = (ctypes.c_void_p,)

    lib.Compile.argtypes = (ctypes.c_void_p,
                            ctypes.c_char_p,
                            ctypes.c_uint,
                            ctypes.c_char_p,
                            ctypes.c_uint,
                            ctypes.c_char_p,
                            ctypes.c_uint,
                            ctypes.c_char_p,
                            ctypes.c_uint,
                            ctypes.POINTER(ctypes.c_uint))
    lib.Compile.restype = ctypes.POINTER(ctypes.c_char)

    lib.ClearPyCompilerProtoString.argtypes = (ctypes.c_void_p,)

    @context_manager_utils.__init__
    def __init__(self):
        self._obj = self.lib.CreatePyCompiler()

    @context_manager_utils.__enter__
    def __enter__(self):
        return self

    @context_manager_utils.__exit__
    def __exit__(self, exc_type, exc_value, traceback):
        self.lib.DeletePyCompiler(self._obj)

    @context_manager_utils.force_in_context
    def compile(self, lgf_pb, hw_spec, sw_config, sim_params):
        spec_data = hw_spec.SerializeToString()
        sw_config_data = sw_config.SerializeToString()
        params_data = sim_params.SerializeToString()
        lgf_data = lgf_pb.SerializeToString()
        bin_out_size = ctypes.c_uint()

        bin_out_ptr = self.lib.Compile(self._obj,
                                       lgf_data,
                                       len(lgf_data),
                                       spec_data,
                                       len(spec_data),
                                       sw_config_data,
                                       len(sw_config_data),
                                       params_data,
                                       len(params_data),
                                       ctypes.byref(bin_out_size))

        bin_out = subgraph_binary_pb2.OPUBinary()
        bin_out.ParseFromString(bin_out_ptr[:bin_out_size.value])

        self.lib.ClearPyCompilerProtoString(self._obj)

        return bin_out


def compile_subgraph(lgf_pb, hw_spec, sw_config, sim_params):
    """Compile the given lgf into a OPUBinary."""
    with PyCompiler() as py_comp:
        bin_out = py_comp.compile(lgf_pb, hw_spec, sw_config, sim_params)

    return bin_out
