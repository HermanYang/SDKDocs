import numpy as np

from sdk2.graph.transform_graph import utils
from sdk2.proto import inference_pb2


def batch_inputs(inputs, batch_size, allow_padding=True):
    """Splits inputs of the same batch dimension size into smaller
    batches of size `batch_size`.

    Args:
        inputs (named_tensor.NamedTensorSet): Input tensors that need to be batched
        batch_size (int): size of each batch
        allow_padding (bool, optional): toggle for padding. Defaults to True.

    Returns:
        inference_pb2.BatchedInferenceInput: Batched Input protobuf
    """

    input_array_coll = list()
    input_edge_coll = list()

    edges = inputs.edges

    for name, tensor in inputs:
        batched_tensors = pad_and_batch_tensor(tensor,
                                               batch_size,
                                               allow_padding=allow_padding)
        input_array_coll.append(batched_tensors)
        input_edge_coll.append(edges[name])

    batched_inputs = inference_pb2.BatchedInferenceInput()

    for arrays in zip(*input_array_coll):
        # EdgeInfo shape gets copied from the array here
        batched_inputs.batches.add().CopyFrom(
            utils.create_inference_inputs(input_edge_coll, arrays))

    return batched_inputs


def pad_and_batch_tensor(tensor, batch_size, allow_padding=True):
    """Pads and batches a tensor according the provided batch size.

    Args:
        tensor (np.ndarray): tensor to be batched
        batch_size (int): size of each batch
        allow_padding (bool, optional): toggle for padding. Only allowed if array size
        is divisible by the batch size. Defaults to True.

    Raises:
        RuntimeError: occurs if padding is disabled and the array size
        isn't divisble by the batch_size.

    Returns:
        List[np.ndarray]: padded and batched tensors.
    """

    shape = tensor.shape

    # Do not allow padding if allow_padding is False
    if not allow_padding and (shape[0] % batch_size != 0):
        raise RuntimeError(
            "Array has {} vectors. Using batch size {}".format(shape, batch_size) +
            " creates padding that may affect calibration")

    # Find padding for total number of batches
    num_batches = np.ceil(shape[0] / batch_size).astype(int)
    pad_batch = (num_batches * batch_size) - shape[0]
    pad = [(0, pad_batch) if i == 0 else (0, 0) for i in range(len(shape))]

    # Split arrays into total number of batches
    padded_tensor = np.pad(tensor, pad, "constant", constant_values=0)
    batched_tensors = np.split(padded_tensor, num_batches, axis=0)

    return batched_tensors
