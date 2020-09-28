import ctypes
import os

from sdk2.proto import common_pb2


class TensorCollection(object):
    """
    Contains a map from strings to common_pb2.Tensor() objects

    Wrapper around a cpp class
    Instances of this class should only be created by a
    graph_collection.GraphCollection() object
    """

    lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__),
                                   "lib_tensor_collection.so"))

    lib.GetTensor.argtypes = (ctypes.c_void_p, ctypes.c_char_p,
                              ctypes.POINTER(ctypes.c_uint))
    lib.GetTensor.restype = ctypes.POINTER(ctypes.c_char)

    lib.ClearProtoString.argtypes = (ctypes.c_void_p,)

    def __init__(self, pointer):
        self._obj = pointer

    def get_tensor(self, key):
        tensor_size = ctypes.c_uint()
        tensor_ptr = self.lib.GetTensor(self._obj, key.encode("utf-8"),
                                        ctypes.byref(tensor_size))

        tensor = common_pb2.Tensor()
        tensor.ParseFromString(tensor_ptr[:tensor_size.value])

        self.lib.ClearProtoString(self._obj)

        return tensor
