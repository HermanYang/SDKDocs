import ctypes
import os

from lt_sdk.common import context_manager_utils
from lt_sdk.graph.graph_collections.collections import (
    histogram_collection,
    simulation_metrics_collection,
    tensor_collection,
    variable_collection,
)


class GraphCollectionBase(object):
    """
    Base class for wrappers around cpp classes
    """
    lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "lib_graph_collection.so"))

    lib.CreateGraphCollection.restype = ctypes.c_void_p
    lib.CreateNullGraphCollection.restype = ctypes.c_void_p
    lib.DeleteGraphCollection.argtypes = (ctypes.c_void_p,)

    lib.GetHistogramCollection.argtypes = (ctypes.c_void_p,)
    lib.GetHistogramCollection.restype = ctypes.c_void_p

    lib.GetVariableCollection.argtypes = (ctypes.c_void_p,)
    lib.GetVariableCollection.restype = ctypes.c_void_p

    lib.GetSimulationMetricsCollection.argtypes = (ctypes.c_void_p,)
    lib.GetSimulationMetricsCollection.restype = ctypes.c_void_p

    lib.GetTensorCollection.argtypes = (ctypes.c_void_p,)
    lib.GetTensorCollection.restype = ctypes.c_void_p

    lib.SerializeToFile.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_char))
    lib.ParseFromFile.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_char))

    def pointer(self):
        return self._obj

    def address(self):
        return self._obj.value

    def is_null(self):
        raise NotImplementedError()


class GraphCollection(GraphCollectionBase):
    """
    A GraphCollection contains a set of objects that can be read, mutated,
    and updated when running a LightGraph.

    Wrapper around a cpp class
    Instances of this class must be created with a context manager
    """

    @context_manager_utils.__init__
    def __init__(self):
        self._obj = ctypes.c_void_p(self.lib.CreateGraphCollection())

        self._hist_coll = histogram_collection.HistogramCollection(
            self.lib.GetHistogramCollection(self._obj))
        self._var_coll = variable_collection.VariableCollection(
            self.lib.GetVariableCollection(self._obj))
        self._sim_metrics_coll = (
            simulation_metrics_collection.SimulationMetricsCollection(
                self.lib.GetSimulationMetricsCollection(self._obj)))
        self._tensor_coll = tensor_collection.TensorCollection(
            self.lib.GetTensorCollection(self._obj))

    @context_manager_utils.__enter__
    def __enter__(self):
        return self

    @context_manager_utils.__exit__
    def __exit__(self, exc_type, exc_value, traceback):
        self.lib.DeleteGraphCollection(self._obj)

    @context_manager_utils.force_in_context
    def is_null(self):
        return False

    @context_manager_utils.force_in_context
    def histogram_collection(self):
        return self._hist_coll

    @context_manager_utils.force_in_context
    def variable_collection(self):
        return self._var_coll

    @context_manager_utils.force_in_context
    def simulation_metrics_collection(self):
        return self._sim_metrics_coll

    @context_manager_utils.force_in_context
    def tensor_collection(self):
        return self._tensor_coll

    @context_manager_utils.force_in_context
    def serialize_to_file(self, fname):
        self.lib.SerializeToFile(self._obj, fname.encode("utf-8"))

    @context_manager_utils.force_in_context
    def parse_from_file(self, fname):
        self.lib.ParseFromFile(self._obj, fname.encode("utf-8"))


class NullGraphCollection(GraphCollectionBase):
    """
    Wrapper around a null cpp class
    Instances of this class DO NOT need to be created within a context manager
    """

    def __init__(self):
        self._obj = self.lib.CreateNullGraphCollection()

    def is_null(self):
        return True
