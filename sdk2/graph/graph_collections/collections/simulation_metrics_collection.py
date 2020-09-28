import ctypes
import os

from sdk2.proto import performance_data_pb2


class SimulationMetricsCollection(object):
    """
    Contains a performance_data_pb2.SimulationMetrics() object

    Wrapper around a cpp class
    Instances of this class should only be created by a
    graph_collection.GraphCollection() object
    """

    lib = ctypes.CDLL(
        os.path.join(os.path.dirname(__file__), "lib_simulation_metrics_collection.so"))

    lib.GetSimulationMetrics.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint))
    lib.GetSimulationMetrics.restype = ctypes.POINTER(ctypes.c_char)

    lib.SetCollectBitActivity.argtypes = (ctypes.c_void_p, ctypes.c_bool)
    lib.SetCollectMemoryLayout.argtypes = (ctypes.c_void_p, ctypes.c_bool)

    lib.InitializeSimulationMetrics.argtypes = (ctypes.c_void_p,
                                                ctypes.POINTER(ctypes.c_char),
                                                ctypes.c_uint)

    lib.ClearProtoString.argtypes = (ctypes.c_void_p,)

    def __init__(self, pointer):
        self._obj = pointer

    def get_simulation_metrics(self):
        sim_metrics_size = ctypes.c_uint()
        sim_metrics_ptr = self.lib.GetSimulationMetrics(self._obj,
                                                        ctypes.byref(sim_metrics_size))

        sim_metrics_pb = performance_data_pb2.SimulationMetrics()
        sim_metrics_pb.ParseFromString(sim_metrics_ptr[:sim_metrics_size.value])

        self.lib.ClearProtoString(self._obj)

        return sim_metrics_pb

    def set_collect_bit_activity(self, collect_bit_activity):
        self.lib.SetCollectBitActivity(self._obj, collect_bit_activity)

    def set_collect_memory_layout(self, collect_memory_layout):
        self.lib.SetCollectMemoryLayout(self._obj, collect_memory_layout)

    def initialize_simulation_metrics(self, hw_specs):
        hw_specs_data = hw_specs.SerializeToString()
        self.lib.InitializeSimulationMetrics(self._obj, hw_specs_data,
                                             len(hw_specs_data))
