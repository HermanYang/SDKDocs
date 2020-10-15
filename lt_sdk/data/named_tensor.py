import copy

import numpy as np

from lt_sdk.proto import dtypes_pb2, lgf_pb2


class NamedTensorSet:

    def __init__(self, names, tensors, dtype=dtypes_pb2.DT_FLOAT):
        if len(names) != len(tensors):
            raise ValueError(
                "names and tensors must have the same length: {}!={}".format(
                    len(names),
                    len(tensors)))

        self.named_tensors = {name: tensor for name, tensor in zip(names, tensors)}

        self._dtype = dtype

    def __getitem__(self, key):
        return self.named_tensors[key]

    def __setitem__(self, key, value):
        assert isinstance(key, str)
        assert isinstance(value, np.ndarray)

        self.named_tensors[key] = value

    def __contains__(self, item):
        return item in self.named_tensors

    def __iter__(self):
        yield from self.named_tensors.items()

    @property
    def names(self):
        return list(self.named_tensors.keys())

    @property
    def ports(self):
        """Returns default ports of edges (can be overloaded).

        Assume port 0 for all edges by default.
        """
        return {name: 0 for name in self.names}

    @property
    def dtypes(self):
        """Returns default dtypes of edges (can be overloaded).

        Assume float32 for all edges by default.
        """
        dtype = dtypes_pb2.DType()
        dtype.t = self._dtype
        dtype.p = 32
        return {name: dtype for name in self.names}

    @property
    def batch_dim_indices(self):
        return {name: 0 for name in self.names}

    @property
    def edges(self):
        """Returns input edge infos without shape."""
        edges = {}
        dtypes = self.dtypes
        ports = self.ports
        batch_dim_indices = self.batch_dim_indices
        for name in self.names:
            edge = lgf_pb2.EdgeInfo()
            edge.name = name
            edge.port = ports[name]
            edge.shape.batch_dim_indx = batch_dim_indices[name]
            edge.dtype.CopyFrom(dtypes[name])
            edges[name] = edge

        return edges

    def apply(self, name, fn, in_place=True):
        """This method applys a function to a specific named tensor.

        Args:
            name ([type]): tensor name
            fn (Callable[[str], np.ndarray]): function to apply to tensor
            in_place (bool, optional): Enables in place operation on current
            NamedTensorSet object if True, otherwise creates a copy. Defaults to True.

        Returns:
            [NamedTensorSet]: Updated NamedTensorSet. Return value can be
            ignored if operation was in place.
        """

        tensor_set = self if in_place else copy.deepcopy(self)
        tensor_set.named_tensors[name] = fn(tensor_set.named_tensors[name])

        return tensor_set

    def apply_all(self, fn, in_place=True):
        """Applies a function to all tensors in a TensorSet

        Args:
            fn (Callable[[str], np.ndarray]): function to apply to tensors
            in_place (bool, optional): Enables in place operation on current
            NamedTensorSet object if True, otherwise creates a copy. Defaults to True.

        Returns:
            [NamedTensorSet]: Updated NamedTensorSet. Return value can be
            ignored if operation was in place.
        """

        tensor_set = self if in_place else copy.deepcopy(self)

        for name in tensor_set.names:
            tensor_set.apply(name, fn)

        return tensor_set
