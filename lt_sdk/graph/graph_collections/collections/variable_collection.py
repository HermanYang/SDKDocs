import ctypes
import os

from lt_sdk.proto import common_pb2


class VariableCollection(object):
    """
    Contains a map from strings to common_pb2.Tensor() objects, these store
    the values of variable nodes in the graph

    Wrapper around a cpp class
    Instances of this class should only be created by a
    graph_collection.GraphCollection() object
    """

    lib = ctypes.CDLL(
        os.path.join(os.path.dirname(__file__),
                     "lib_variable_collection.so"))

    lib.GetVariable.argtypes = (ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_char),
                                ctypes.POINTER(ctypes.c_uint))
    lib.GetVariable.restype = ctypes.POINTER(ctypes.c_char)

    lib.AddVariable.argtypes = (ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_char),
                                ctypes.POINTER(ctypes.c_char),
                                ctypes.c_uint)

    lib.VariableCollectionGetKeys.argtypes = (ctypes.c_void_p,
                                              ctypes.POINTER(ctypes.c_uint))
    lib.VariableCollectionGetKeys.restype = ctypes.POINTER(ctypes.c_char)

    lib.ClearProtoString.argtypes = (ctypes.c_void_p,)

    def __init__(self, pointer):
        self._obj = pointer

    def get_variable(self, node_name):
        tensor_size = ctypes.c_uint()
        tensor_ptr = self.lib.GetVariable(self._obj,
                                          node_name.encode("utf-8"),
                                          ctypes.byref(tensor_size))

        tensor_pb = common_pb2.Tensor()
        tensor_pb.ParseFromString(tensor_ptr[:tensor_size.value])

        self.lib.ClearProtoString(self._obj)

        return tensor_pb

    def add_variable(self, node_name, tensor_pb):
        tensor_data = tensor_pb.SerializeToString()
        self.lib.AddVariable(self._obj,
                             node_name.encode("utf-8"),
                             tensor_data,
                             len(tensor_data))

    def get_keys(self):
        keys_size = ctypes.c_uint()
        keys_ptr = self.lib.VariableCollectionGetKeys(self._obj, ctypes.byref(keys_size))

        keys = common_pb2.Param()
        keys.ParseFromString(keys_ptr[:keys_size.value])

        self.lib.ClearProtoString(self._obj)

        return list(keys.l.s)
