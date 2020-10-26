import logging
import multiprocessing

import numpy as np
import tensorflow as tf

from lt_sdk.proto import common_pb2, dtypes_pb2, inference_pb2, lgf_pb2

LT_UNSET = "_LT_UNSET_:0"
SCALE_EPS_VALUE = np.finfo(np.float64).eps

NP_TYPES = {
    np.dtype(np.float32): (dtypes_pb2.DT_FLOAT,
                           32),
    np.dtype(np.float64): (dtypes_pb2.DT_FLOAT,
                           32),
    np.dtype(np.int8): (dtypes_pb2.DT_INT,
                        8),
    np.dtype(np.int16): (dtypes_pb2.DT_INT,
                         16),
    np.dtype(np.int32): (dtypes_pb2.DT_INT,
                         32),
    np.dtype(np.int64): (dtypes_pb2.DT_INT,
                         64),
    np.dtype(np.uint8): (dtypes_pb2.DT_UINT,
                         8),
    np.dtype(np.uint16): (dtypes_pb2.DT_UINT,
                          16),
    np.dtype(np.bool): (dtypes_pb2.DT_BOOL,
                        1),
    np.dtype(np.complex64): (dtypes_pb2.DT_COMPLEX,
                             64),
    np.dtype(np.complex128): (dtypes_pb2.DT_COMPLEX,
                              128),
}


def log_message(msg):
    logging.info("---- {0} ----".format(msg))


def dtype_pb_to_np_dtype(lgf_dtype):
    """Converts a dtypes_pb.DType() to a numpy dtype"""
    if lgf_dtype.t == dtypes_pb2.DT_FLOAT:
        if lgf_dtype.p <= 16:
            return np.float16
        elif lgf_dtype.p <= 32:
            return np.float32
        else:
            return np.float64

    elif lgf_dtype.t == dtypes_pb2.DT_BFLOAT:
        if lgf_dtype.p <= 16:
            return tf.bfloat16.as_numpy_dtype
        else:
            return np.float32

    elif lgf_dtype.t in [dtypes_pb2.DT_INT, dtypes_pb2.DT_QINT]:
        if lgf_dtype.p <= 8:
            return np.int8
        elif lgf_dtype.p <= 16:
            return np.int16
        elif lgf_dtype.p <= 32:
            return np.int32
        else:
            return np.int64

    elif lgf_dtype.t in [dtypes_pb2.DT_UINT, dtypes_pb2.DT_QUINT]:
        if lgf_dtype.p <= 8:
            return np.uint8
        elif lgf_dtype.p <= 16:
            return np.uint16
        elif lgf_dtype.p <= 32:
            return np.uint32
        else:
            return np.uint64

    elif lgf_dtype.t == dtypes_pb2.DT_COMPLEX:
        if lgf_dtype.p <= 64:
            return np.complex64
        else:
            return np.complex128

    elif lgf_dtype.t == dtypes_pb2.DT_BOOL:
        return np.bool

    elif lgf_dtype.t == dtypes_pb2.DT_STRING:
        return np.str

    else:
        raise ValueError("Cannot convert {} {} to numpy dtype".format(
            dtypes_pb2.Type.Name(lgf_dtype.t),
            lgf_dtype.p))


def np_dtype_to_lgf_dtype(np_dtype):
    """Converts a numpy dtype to a dtypes_pb.DType()."""
    t, p = NP_TYPES.get(np_dtype, (dtypes_pb2.DT_INVALID, 0))
    ret = dtypes_pb2.DType()
    ret.t = t
    ret.p = p
    return ret


def np_to_edge_info(np_array, name=LT_UNSET):
    """Use shape and dtype from an np array to get edge info."""
    ret = lgf_pb2.EdgeInfo()
    ret.dtype.CopyFrom(np_dtype_to_lgf_dtype(np_array.dtype))
    ret.shape.d.extend(np_array.shape)
    ret.name = name
    return ret


def array_to_tensor_pb(array, dtype_pb, batch_dim_indx=-1):
    """
    Params:
        array: numpy array
        dtype_pb: dtypes_pb2.DType

    Returns:
        tensor_pb: a common_pb2.Tensor(), the shape and contents are
            the same as array casted to the provided dtype
    """
    if array.size == 1 and len(array.shape) == 0:
        array = array.reshape([1])

    tensor_pb = common_pb2.Tensor()
    tensor_pb.dtype.CopyFrom(dtype_pb)
    tensor_pb.shape.d.extend(array.shape)
    tensor_pb.shape.batch_dim_indx = batch_dim_indx
    try:
        tensor_pb.tensor_content = array.astype(
            dtype_pb_to_np_dtype(dtype_pb)).tostring()
    except ValueError as e:
        logging.error(array)
        logging.error("casting {0} to {1}".format(array.dtype,
                                                  dtype_pb_to_np_dtype(dtype_pb)))
        raise e

    return tensor_pb


def tensor_pb_to_array(tensor_pb, dtype_np):
    """
    Params:
        tensor_pb: a common_pb2.Tensor()
        dtype_np: a numpy dtype

    Returns:
        array: a numpy array, the shape and contents are the same as
            as tensor_pb casted to the provided dtype
    """
    array = np.frombuffer(tensor_pb.tensor_content,
                          dtype=dtype_pb_to_np_dtype(tensor_pb.dtype))
    array = array.reshape(tensor_pb.shape.d).astype(dtype_np)

    return array


def create_inference_inputs(edge_list, array_list):
    """
    Params:
        edge_list: a list of lgf_pb2.EdgeInfo() protobufs
        array_list: a list of numpy arrays corresponding to edge_list

    Returns
        inputs: a inference_pb2.InferenceInput() protobuf
    """
    if len(edge_list) != len(array_list):
        raise ValueError("Edge list does not match array list: {0} != {1}".format(
            edge_list,
            array_list))
    inputs = inference_pb2.InferenceInput()
    for i, e in enumerate(edge_list):
        named_tensor = inputs.inputs.add()
        named_tensor.data.CopyFrom(array_to_tensor_pb(array_list[i], e.dtype))
        named_tensor.data.shape.batch_dim_indx = e.shape.batch_dim_indx
        named_tensor.data.shape.batch_dilation_factor = e.shape.batch_dilation_factor

        named_tensor.edge_info.CopyFrom(e)
        # edge shape may contain missing dims
        named_tensor.edge_info.shape.CopyFrom(named_tensor.data.shape)

    return inputs


def generate_random_inference_inputs(edge_list, unknown_dim_size=1):
    array_list = []
    for e in edge_list:
        # Remove missing dims from the shape. Assume all dims with value
        # -1 are batch dimensions.
        shape = [d if d != -1 else unknown_dim_size for d in e.shape.d]
        array_list.append(np.random.rand(*shape))

    return create_inference_inputs(edge_list, array_list)


def get_min_and_max_of_precision(precision, signed=True):
    min_val = -(1 << (precision - 1))
    max_val = (1 << (precision - 1)) - 1

    if not signed:
        min_val = 0
        max_val = (1 << precision) - 1

    return min_val, max_val


def get_quant_scale_from_quant_range(quant_range, precision):
    """
    Params:
        quant_range: a tuple (low, high)
        precision: quantizatin precision

    Returns:
        quantization scale
    """
    low, high = quant_range
    quant_scale = (high - low) / ((1 << precision) - 1)

    if quant_scale < SCALE_EPS_VALUE:
        logging.warning(
            "The calculated scale is too small. Set it to {}".format(SCALE_EPS_VALUE))
        return SCALE_EPS_VALUE

    return quant_scale


def get_corrected_symm_quant_range(ll, precision):
    """Calculate the corrected range for symmetric quantization.

    This is to account for the positive range being a bit smaller than
    the negative range:
    high = |low| * corr
    """
    assert ll < 0
    corr = 1 - 1 / (1 << (precision - 1))
    return (ll, -ll * corr)


def get_corrected_asymm_quant_range(low, high, precision):
    """Calculate the corrected range for asymmetric quantization"""
    zero_point = (high + low) / 2
    sym_low = low - zero_point
    sym_low, sym_high = get_corrected_symm_quant_range(sym_low, precision)
    return (sym_low + zero_point, sym_high + zero_point)


def quantize_array(array, precision, quant_scale=None, quant_range=None):
    """
    Params:
        array: numpy array
        precision: quantization precision
        quant_scale: quantization scale, must be provided if quant_range is None
        quant_range: a tuple (low, high), must be provided if quant_scale is None
    """
    if quant_scale is None:
        assert quant_range is not None
        quant_scale = get_quant_scale_from_quant_range(quant_range, precision)

    quant_min, quant_max = get_min_and_max_of_precision(precision)
    return np.clip(np.round(array / quant_scale), quant_min, quant_max)


def array_to_phase_incoherent(normalized_array):
    """
    params:
        normalized_array: numpy array to be converted to phases, must be in range [-1, 1]
    returns:
        phases: numpy array of phases
    """
    assert (np.max(np.abs(normalized_array)) <= 1)

    # arcsin range is [-pi/2, pi/2], arccos range is [0, pi]
    return -np.arcsin(normalized_array)


def run_fn_with_multiprocessing(args_list, fn, num_processes=1):
    """
    Params:
        args_list: a list of args where each args object is a tuple
        fn: a function to call that takes args and returns a single object

    Returns:
        results: a list such that results[i] == fn(args)
    """
    results = []
    if num_processes == 1 or len(args_list) == 1:
        for args in args_list:
            results.append(fn(args))
    else:
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            results.extend(p.map(fn, args_list))

    return results


def edges_match(e1, e2, check_consistent=False):
    # Edges are uniquely identified by name and port
    edges_match = (e1.name == e2.name) and (e1.port == e2.port)

    # check_consistent also requires dtypes and shapes to match
    if check_consistent:
        edges_match = edges_match and (e1.dtype == e2.dtype) and (e1.shape == e2.shape)

    return edges_match
