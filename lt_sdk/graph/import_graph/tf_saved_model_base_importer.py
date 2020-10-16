import logging
import re

import tensorflow as tf
from google.protobuf.pyext import _message
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.saved_model import signature_constants

from lt_sdk.graph import lgf_graph
from lt_sdk.graph.import_graph import graph_importer
from lt_sdk.graph.transform_graph import utils
from lt_sdk.proto import common_pb2, dtypes_pb2, graph_types_pb2, lgf_pb2, ops_pb2


class ImportTFSavedModelBase(graph_importer.ImportGraph):
    """Interface to import a Tensorflow Saved Model"""

    PARTIAL_META_GRAPH_DEF = "partial_meta_graph_def"
    TENSOR_PB = "tensor_pb"
    OUTPUT_NODES_COLLECTION_DEF_KEY = "lt_output_nodes"

    RUN_1_BATCH_SIZE = 1
    RUN_2_BATCH_SIZE = 2

    def __init__(self,
                 *args,
                 input_names=None,
                 output_names=None,
                 allow_multiple_batch_dims=False,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self._meta_graph_def = None
        self._input_tensor_names = input_names
        self._output_tensor_names = output_names
        self._output_node_names = None
        self._required_nodes = set()
        self._allow_multiple_batch_dims = allow_multiple_batch_dims

        self.load_graph_session(self.read_graph)

    DATA_TYPE_MAP = {
        tf.float32: (dtypes_pb2.DT_FLOAT,
                     32),
        tf.float64: (dtypes_pb2.DT_FLOAT,
                     64),
        tf.bfloat16: (dtypes_pb2.DT_BFLOAT,
                      16),
        tf.int8: (dtypes_pb2.DT_INT,
                  8),
        tf.int16: (dtypes_pb2.DT_INT,
                   16),
        tf.int32: (dtypes_pb2.DT_INT,
                   32),
        tf.int64: (dtypes_pb2.DT_INT,
                   64),
        tf.uint8: (dtypes_pb2.DT_UINT,
                   8),
        tf.uint16: (dtypes_pb2.DT_UINT,
                    16),
        tf.string: (dtypes_pb2.DT_STRING,
                    0),
        tf.bool: (dtypes_pb2.DT_BOOL,
                  1),
        tf.complex64: (dtypes_pb2.DT_COMPLEX,
                       64),
        tf.complex128: (dtypes_pb2.DT_COMPLEX,
                        128),
        tf.qint8: (dtypes_pb2.DT_QINT,
                   8),
        tf.qint16: (dtypes_pb2.DT_QINT,
                    16),
        tf.qint32: (dtypes_pb2.DT_QINT,
                    32),
        tf.quint8: (dtypes_pb2.DT_QUINT,
                    8),
        tf.quint16: (dtypes_pb2.DT_QUINT,
                     16),
    }

    REV_DATA_TYPE_MAP = {
        (dtypes_pb2.DT_FLOAT,
         32): tf.float32,
        (dtypes_pb2.DT_FLOAT,
         64): tf.float64,
        (dtypes_pb2.DT_BFLOAT,
         16): tf.bfloat16,
        (dtypes_pb2.DT_INT,
         8): tf.int8,
        (dtypes_pb2.DT_INT,
         16): tf.int16,
        (dtypes_pb2.DT_INT,
         32): tf.int32,
        (dtypes_pb2.DT_INT,
         64): tf.int64,
        (dtypes_pb2.DT_UINT,
         8): tf.uint8,
        (dtypes_pb2.DT_UINT,
         16): tf.uint16,
        (dtypes_pb2.DT_STRING,
         0): tf.string,
        (dtypes_pb2.DT_BOOL,
         1): tf.bool,
        (dtypes_pb2.DT_COMPLEX,
         64): tf.complex64,
        (dtypes_pb2.DT_COMPLEX,
         128): tf.complex128,
        (dtypes_pb2.DT_QINT,
         8): tf.qint8,
        (dtypes_pb2.DT_QINT,
         16): tf.qint16,
        (dtypes_pb2.DT_QINT,
         32): tf.qint32,
        (dtypes_pb2.DT_QUINT,
         8): tf.quint8,
        (dtypes_pb2.DT_QUINT,
         16): tf.quint16
    }

    OP_MAP = {
        "Conv2D": ops_pb2.CONV2D,
        "MatMul": ops_pb2.MATMUL,
        "SparseMatMul": ops_pb2.MATMUL,
        "DepthwiseConv2dNative": ops_pb2.DEPTHWISE_CONV2D,
        "BiasAdd": ops_pb2.ADD,
        "Add": ops_pb2.ADD,
        "AddV2": ops_pb2.ADD,
        "Const": ops_pb2.CONST,
        "Relu": ops_pb2.RELU,
        "Relu6": ops_pb2.RELU6,
        "Reshape": ops_pb2.RESHAPE,
        "AvgPool": ops_pb2.AVGPOOL,
        "MaxPool": ops_pb2.MAXPOOL,
        "Identity": ops_pb2.IDENTITY,
        "Enter": ops_pb2.ENTER,
        "Switch": ops_pb2.SWITCH,
        "Merge": ops_pb2.MERGE,
        "NextIteration": ops_pb2.NEXT_ITERATION,
        "Exit": ops_pb2.EXIT,
        "FusedBatchNorm": ops_pb2.BATCHNORM,
        "FusedBatchNormV2": ops_pb2.BATCHNORM,
        "FusedBatchNormV3": ops_pb2.BATCHNORM,
        "Pad": ops_pb2.PAD,
        "Mean": ops_pb2.MEAN,
        "Softmax": ops_pb2.SOFTMAX,
        "VariableV2": ops_pb2.VARIABLE,
        "Assign": ops_pb2.ASSIGN,
        "Squeeze": ops_pb2.SQUEEZE,
        "swish_f32": ops_pb2.SWISH,
        "Mul": ops_pb2.MULTIPLY,
        "Sigmoid": ops_pb2.SIGMOID,
        "Tanh": ops_pb2.TANH,
        "Transpose": ops_pb2.TRANSPOSE,
        "Unpack": ops_pb2.UNSTACK,
        "Sub": ops_pb2.SUB,
        "Pow": ops_pb2.POW,
        "ExpandDims": ops_pb2.EXPANDDIMS,
        "Fill": ops_pb2.FILL,
        "Rsqrt": ops_pb2.RSQRT,
        "SquaredDifference": ops_pb2.SQUARED_DIFFERENCE,
        "Pack": ops_pb2.STACK,
        "BatchMatMulV2": ops_pb2.BATCHMATMUL,
        "Tile": ops_pb2.TILE,
        "ConcatV2": ops_pb2.CONCAT,
        "Split": ops_pb2.SPLIT,
        "SplitV": ops_pb2.SPLIT,
    }

    @staticmethod
    def get_node_name_and_output_index(name):
        """Strips off ports and other decorations to get the underlying node name.

        The name from the input of a NodeDef object might include a port number in the
        form of ":1", ":2", etc. Under some circumstances it might start with a "^".
        Args:
            name: a string represents the name of a node or a tensor.
                Possibily from the input of a NodeDef object.

        Returns:
            A tuple in the formed of (stripped_node_name, output_index, is_control)
        """
        is_control = False
        if name.startswith("^"):
            name = name[1:]
            is_control = True

        op_name, index = name, 0
        if re.search(".*:[0-9]+", op_name):
            op_name, index = op_name.rsplit(":", 1)
            index = int(index)

        return op_name, index, is_control

    @staticmethod
    def get_input_tensor_names_from_meta_graph_def(meta_graph_def):
        input_names = set()
        default_sig = meta_graph_def.signature_def[
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        for _, v in default_sig.inputs.items():
            input_names.add(v.name)

        return list(input_names)

    @staticmethod
    def get_output_tensor_names_from_meta_graph_def(meta_graph_def):
        output_names = set()
        default_sig = meta_graph_def.signature_def[
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        for _, v in default_sig.outputs.items():
            output_names.add(v.name)

        return list(output_names)

    @staticmethod
    def get_output_node_names_from_meta_graph_def(meta_graph_def):
        output_node_names = set()
        if (ImportTFSavedModelBase.OUTPUT_NODES_COLLECTION_DEF_KEY
                in meta_graph_def.collection_def):
            output_node_names.update(meta_graph_def.collection_def[
                ImportTFSavedModelBase.OUTPUT_NODES_COLLECTION_DEF_KEY].node_list.value)

        return output_node_names

    def _add_batch_dim_indx(self, lgf_tensor_shape):
        """Find batch dim in the given tensor shape.

        Batch dim is defined to be the dim with negative size.
        In most cases, there is only one batch dim in a tensor. When there are multiple
        negative dimensions, we fall back to setting batch_dim_indx to 0 if dim 0 is
        negative. Otherwise, we either set batch_dim_indx to the first negative dimension
        (when self._allow_multiple_batch_dims is True) or raise an exception.
        """
        for i, d in enumerate(lgf_tensor_shape.d):
            if d < 0 and lgf_tensor_shape.batch_dim_indx != i:
                if lgf_tensor_shape.batch_dim_indx == -1:
                    lgf_tensor_shape.batch_dim_indx = i
                elif lgf_tensor_shape.d[0] < 0:
                    lgf_tensor_shape.batch_dim_indx = 0
                    logging.warning(
                        "Multiple batch dims found: %d, %d, set batch_dim_indx to 0",
                        lgf_tensor_shape.batch_dim_indx,
                        i)
                    return
                elif self._allow_multiple_batch_dims:
                    logging.warning("Multiple batch dims found: %d, %d",
                                    lgf_tensor_shape.batch_dim_indx,
                                    i)
                else:
                    raise ValueError("Multiple batch dims found in TensorShape %s" %
                                     lgf_tensor_shape)

    def tf_tensorshape_to_lgf_tensorshape(self,
                                          tf_tensorshape,
                                          update_batch_dim_indx=True):
        lgf_tensor_shape = common_pb2.TensorShape()
        lgf_tensor_shape.batch_dim_indx = -1
        if tf_tensorshape == tf.TensorShape(None):
            return lgf_tensor_shape
        elif tf_tensorshape == tf.TensorShape([]):
            return lgf_tensor_shape

        lgf_tensor_shape.d.extend(
            [-1 if d.value is None else int(d.value) for d in tf_tensorshape.dims])

        if update_batch_dim_indx:
            self._add_batch_dim_indx(lgf_tensor_shape)

        return lgf_tensor_shape

    def np_shape_to_lgf_tensorshape(self, np_shape, update_batch_dim_indx=True):
        lgf_tensor_shape = common_pb2.TensorShape()
        lgf_tensor_shape.batch_dim_indx = -1
        if len(np_shape) == 0:
            return lgf_tensor_shape

        lgf_tensor_shape.d.extend([-1 if d is None else int(d) for d in np_shape])

        if update_batch_dim_indx:
            self._add_batch_dim_indx(lgf_tensor_shape)

        return lgf_tensor_shape

    @staticmethod
    def tf_dtype_to_lgf_dtype(tf_dtype):
        lgf_dtype = dtypes_pb2.DType()
        if tf_dtype.base_dtype in ImportTFSavedModelBase.DATA_TYPE_MAP:
            t, p = ImportTFSavedModelBase.DATA_TYPE_MAP[tf_dtype.base_dtype]
            lgf_dtype.t = t
            lgf_dtype.p = p
        return lgf_dtype

    @staticmethod
    def get_strings_from_proto(proto):
        # Gets all strings from a protobuf
        strings = []
        for _, val in proto.ListFields():
            # val is a string
            if isinstance(val, str):
                strings.append(val)

            # val is a list of singular values
            elif isinstance(val, _message.RepeatedScalarContainer):
                strings.extend([v for v in val if isinstance(v, str)])

            # val is a list of protobufs
            elif isinstance(val, _message.RepeatedCompositeContainer):
                for v in val:
                    strings.extend(ImportTFSavedModelBase.get_strings_from_proto(v))

            # val is a map
            elif isinstance(val, _message.MessageMapContainer):
                for k, v in val.items():
                    # keys in the map are strings
                    if isinstance(k, str):
                        strings.append(k)

                    # values in the map are strings
                    if isinstance(val, str):
                        strings.append(v)

                    # values in the map are protobufs
                    elif hasattr(v, "ListFields"):
                        strings.extend(ImportTFSavedModelBase.get_strings_from_proto(v))

            # val is a protobuf
            elif hasattr(val, "ListFields"):
                strings.extend(ImportTFSavedModelBase.get_strings_from_proto(val))

        return strings

    def load_graph_session(self, read_graph_fn):
        """
        Loads a graph into a session, sets self._meta_graph_def to a meta_graph_def
        representative of the session, then calls read_graph_fn(sess) within the
        execution context of the session
        """
        raise NotImplementedError()

    def _get_feed_dict(self, sess, unknown_dim_size):
        # Create feed dict with random data
        feed_dict = {}

        for inp in self._input_tensor_names:
            real_name, port, _ = self.get_node_name_and_output_index(inp)
            # This shape might have multiple unknown dimensions, so we
            # don't update batch_dim_indx here
            shape = self.tf_tensorshape_to_lgf_tensorshape(
                sess.graph.get_tensor_by_name(inp).shape,
                update_batch_dim_indx=False)

            # Correct for batch size if necessary
            corrected_edge_info = lgf_pb2.EdgeInfo()
            corrected_edge_info.CopyFrom(self._input_edges[(real_name, port)])
            batch_dim_indx = corrected_edge_info.shape.batch_dim_indx
            if (batch_dim_indx >= 0 and len(shape.d) > 0
                    and shape.d[batch_dim_indx] > 0):
                corrected_edge_info.shape.d[batch_dim_indx] = shape.d[batch_dim_indx]

            data = utils.generate_random_inference_inputs(
                [corrected_edge_info],
                unknown_dim_size=unknown_dim_size)
            named_tensor = data.inputs[0]
            feed_dict[sess.graph.get_tensor_by_name(inp)] = utils.tensor_pb_to_array(
                named_tensor.data,
                utils.dtype_pb_to_np_dtype(named_tensor.data.dtype))

        return feed_dict

    def _get_fetches(self, sess):
        # Get the subgraph of nodes necessary to run the output ops
        output_node_names = [
            self.get_node_name_and_output_index(ten_name)[0]
            for ten_name in self._output_tensor_names
        ]
        subgraph = tf.graph_util.extract_sub_graph(sess.graph_def, output_node_names)
        subgraph_node_names = {n.name for n in subgraph.node}

        # Get fetches
        fetches = set()
        if self._input_tensor_names:
            fetches = fetches.union(set(self._input_tensor_names))
        if self._output_tensor_names:
            fetches = fetches.union(set(self._output_tensor_names))
        for op in sess.graph.get_operations():
            # Skip ops that the output does not depend on
            if op.node_def.name not in subgraph_node_names:
                continue

            # Go through inputs of the op, TF seems to ignore output tensors of
            # an op if no other node in the graph asks for that tensor
            op_lnf = self._init_lnf_from_tf_node_def(op.node_def)
            ignore_op = self._ignore_nodes_filter.matches(op_lnf,
                                                          lgf_graph.LightGraph([op_lnf]))

            for inp in op.inputs:
                # Skip constant tensors
                if (inp.op.type == "Const"):
                    continue

                # If we have (ignore) --> (ignore), we can skip
                inp_lnf = self._init_lnf_from_tf_node_def(inp.op.node_def)
                ignore_inp = self._ignore_nodes_filter.matches(
                    inp_lnf,
                    lgf_graph.LightGraph([inp_lnf]))
                if ignore_op and ignore_inp:
                    continue

                # Otherwise, we will need the tensor because we need
                # inputs and outputs of all (not ignore) nodes. Cases here are
                # (not ignore) --> (not ignore), (ignore) --> (not ignore),
                # (not ignore) --> (ignore)
                fetches.add(inp.name)

        return list(fetches)

    def _update_batch_dim(self, ten_name, shape_1, shape_2):
        assert len(shape_1) == len(shape_2)

        if len(shape_1) == 0:
            return

        lgf_tensorshape = self._tensor_shapes[ten_name]
        # Check for batch dim. Batch dims do not match and should have values
        # 1 and 2 respectively. The other dims are expected to match. Otherwise
        # a warning is emitted.
        for i, (s1, s2) in enumerate(zip(shape_1, shape_2)):
            if s1 != s2:
                dilation_factor = s1 / self.RUN_1_BATCH_SIZE
                if (dilation_factor == s2 / self.RUN_2_BATCH_SIZE
                        and dilation_factor == int(dilation_factor)):
                    if lgf_tensorshape.batch_dilation_factor == 0:
                        lgf_tensorshape.batch_dilation_factor = int(dilation_factor)
                        lgf_tensorshape.d[i] = -1
                    else:
                        assert lgf_tensorshape.batch_dilation_factor == dilation_factor
                else:
                    logging.warning(
                        "Found non-matching dimension %d for tensor %s: %d, %d",
                        i,
                        ten_name,
                        s1,
                        s2)
        self._add_batch_dim_indx(lgf_tensorshape)

    def _get_tf_tensorshape(self, tf_ten):
        if tf_ten.op.type == "Const":
            value = tf.make_ndarray(tf_ten.op.node_def.attr["value"].tensor)
            if value.size == 1 and len(value.shape) == 0:
                return tf.TensorShape([1])

        return tf_ten.shape

    def read_graph(self, sess):
        # These should be set in load_graph_session before calling this function
        assert (self._meta_graph_def is not None)
        assert (self._input_tensor_names is not None)
        assert (self._output_tensor_names is not None)

        # Get some things from the session
        ops = sess.graph.get_operations()
        variable_map = {
            self.get_node_name_and_output_index(v.name)[0]: v
            for v in (tf.global_variables() + tf.local_variables() +
                      tf.trainable_variables() + tf.model_variables())
        }

        # Initialize some variables
        self._graph_def = sess.graph_def
        self._max_ports = {}
        self._variable_values = {}
        self._tensor_shapes = {}
        self._tensor_dtypes = {}

        # Initialize tensorflow variables if necesssary
        for var in variable_map.values():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                logging.warning("Encountered uninitialized variable: {0}".format(
                    var.name))
                sess.run(var.initializer)

        # Run random data through the graph to determine shapes of
        # some of the tensors
        if self._input_edges:
            # Create two feed dicts with different batch sizes
            feed_dict_batch_1 = self._get_feed_dict(sess, self.RUN_1_BATCH_SIZE)
            feed_dict_batch_2 = self._get_feed_dict(sess, self.RUN_2_BATCH_SIZE)
            fetches = self._get_fetches(sess)

            # Run tensorflow on the fetches
            try:
                arrays_1 = sess.run(fetches, feed_dict=feed_dict_batch_1)
                arrays_2 = sess.run(fetches, feed_dict=feed_dict_batch_2)
            except tf.errors.InternalError:
                # Internal error usually means we asked for a tensor in a loop,
                # which is a no-no. Try just the output tensors and let the
                # default logic below handle the rest.
                logging.error("Error finding tensor shapes, resorting to outputs")
                arrays_1 = sess.run(self._output_tensor_names,
                                    feed_dict=feed_dict_batch_1)
                arrays_2 = sess.run(self._output_tensor_names,
                                    feed_dict=feed_dict_batch_2)

            # Fill in tensor shapes and dtypes
            for i in range(len(arrays_1)):
                array_1 = arrays_1[i]
                array_2 = arrays_2[i]
                ten_name = fetches[i]

                self._tensor_shapes[ten_name] = self.np_shape_to_lgf_tensorshape(
                    array_1.shape,
                    update_batch_dim_indx=False)
                self._tensor_dtypes[ten_name] = utils.np_dtype_to_lgf_dtype(
                    array_1.dtype)

                self._update_batch_dim(ten_name, array_1.shape, array_2.shape)

        # Get shapes and dtypes for the rest of the ops and variables
        for op in ops:
            self._max_ports[op.name] = len(op.outputs)

            # The op is a variable, store its values in self._variable_values
            if self.OP_MAP.get(op.type, ops_pb2.UNKNOWN) == ops_pb2.VARIABLE:
                var = variable_map[op.name]
                self._variable_values[op.name] = var.value().eval(session=sess)
                shape_proto = op.get_attr("shape")
                self._tensor_shapes[var.name] = self.tf_tensorshape_to_lgf_tensorshape(
                    tf.TensorShape([d.size for d in shape_proto.dim]))
                self._tensor_dtypes[var.name] = self.tf_dtype_to_lgf_dtype(
                    op.get_attr("dtype"))

            # Op does not have any outputs, use tensor info to get the shape
            elif not len(op.outputs):
                try:
                    ten = sess.graph.get_tensor_by_name(op.name + ":0")
                    self._tensor_shapes[
                        ten.name] = self.tf_tensorshape_to_lgf_tensorshape(ten.shape)
                    self._tensor_dtypes[ten.name] = self.tf_dtype_to_lgf_dtype(ten.dtype)
                except KeyError:
                    pass

            # All other cases
            for out in op.outputs:
                if out.name not in self._tensor_shapes:
                    self._tensor_shapes[
                        out.name] = self.tf_tensorshape_to_lgf_tensorshape(
                            self._get_tf_tensorshape(out))
                    self._tensor_dtypes[out.name] = self.tf_dtype_to_lgf_dtype(out.dtype)

    def _tensor_name_to_edge_info(self, tensor_name):
        edge_info = lgf_pb2.EdgeInfo()
        name, port, _ = self.get_node_name_and_output_index(tensor_name)
        edge_info.name = name
        edge_info.port = port
        edge_info.dtype.CopyFrom(self._tensor_dtypes[tensor_name])
        edge_info.shape.CopyFrom(self._tensor_shapes[tensor_name])
        return edge_info

    def _tensor_names_from_tf_node_def_input(self, tf_node_def):
        # Returns a list of the inputs to the tf_node_def, where
        # the inputs are formatted as tensor names
        tensor_names = []
        controls = []
        for inp_name in tf_node_def.input:
            name, port, is_control = self.get_node_name_and_output_index(inp_name)
            tensor_name = "{}:{}".format(name, port)
            tensor_names.append(tensor_name)
            controls.append(is_control)

        return tensor_names, controls

    @staticmethod
    def _init_lnf_from_tf_node_def(tf_node_def):
        lnf = lgf_pb2.LNF()
        lnf.name = tf_node_def.name

        # Node attributes
        lnf.supported = False
        if tf_node_def.op == "LGFSubgraph":
            # Subgraph node
            lnf.subgraph.SetInParent()
            lnf.subgraph.graph.ParseFromString(tf_node_def.attr["serialized_subgraph"].s)
        else:
            # Original node
            lnf.original.SetInParent()
            lnf.original.t = graph_types_pb2.TFSavedModel
            lnf.original.op = ImportTFSavedModelBase.OP_MAP.get(
                tf_node_def.op,
                ops_pb2.UNKNOWN)

            if lnf.original.op == ops_pb2.ENTER:
                lnf.original.attr[lgf_graph.LightGraph.
                                  IS_CONST_ATTR].b = tf_node_def.attr["is_constant"].b

            lnf.original.serialized_node = tf_node_def.SerializeToString()

        return lnf

    def _tf_node_def_to_lnf(self, tf_node_def):
        lnf = self._init_lnf_from_tf_node_def(tf_node_def)

        # Node inputs
        input_tensor_names, is_controls = self._tensor_names_from_tf_node_def_input(
            tf_node_def)
        for i, tensor_name in enumerate(input_tensor_names):
            if is_controls[i]:
                name, port, _ = self.get_node_name_and_output_index(tensor_name)
                lnf.control_inputs.append(name)
            else:
                lnf.inputs.add().CopyFrom(self._tensor_name_to_edge_info(tensor_name))

        # Node outputs
        max_port = self._max_ports[lnf.name]

        output_tensor_names = [
            "{0}:{1}".format(lnf.name,
                             port) for port in range(max_port)
        ]
        lnf.outputs.extend([
            self._tensor_name_to_edge_info(tensor_name)
            for tensor_name in output_tensor_names
        ])

        # Node attributes
        if lnf.HasField(lgf_pb2.LNF.subgraph.DESCRIPTOR.name):
            del (lnf.outputs[-1])
        if lnf.original.op == ops_pb2.VARIABLE:
            tensor_pb = tf.make_tensor_proto(
                self._variable_values[lnf.name],
                dtype=tf.dtypes.as_dtype(
                    self._variable_values[lnf.name].dtype).as_datatype_enum)
            lnf.original.attr[self.TENSOR_PB].v = tensor_pb.SerializeToString()

        return lnf

    def _tf_meta_graph_def_to_lgf_meta_graph_info(self, meta_graph_def):
        # Store the serialized partial meta graph def
        partial_meta_graph_def = tf.MetaGraphDef()
        partial_meta_graph_def.saver_def.CopyFrom(meta_graph_def.saver_def)
        for k, v in meta_graph_def.collection_def.items():
            partial_meta_graph_def.collection_def[k].CopyFrom(v)
        for k, v in meta_graph_def.signature_def.items():
            partial_meta_graph_def.signature_def[k].CopyFrom(v)

        partial_meta_graph_def.graph_def.CopyFrom(meta_graph_def.graph_def)
        del (partial_meta_graph_def.graph_def.node[:])

        meta_graph_info = lgf_pb2.MetaGraphInfo()
        meta_graph_info.original_graph_info[
            ImportTFSavedModelBase.
            PARTIAL_META_GRAPH_DEF].v = partial_meta_graph_def.SerializeToString()

        # Get all strings from the partial meta graph def
        proto_strings = self.get_strings_from_proto(partial_meta_graph_def)

        # Need to manually deserialize the collection defs
        for k, collection_def in partial_meta_graph_def.collection_def.items():
            if collection_def.HasField("bytes_list"):
                proto_type = tf_ops.get_collection_proto_type(k)
                for serialized_proto in collection_def.bytes_list.value:
                    proto = proto_type()
                    proto.ParseFromString(serialized_proto)
                    proto_strings.extend(self.get_strings_from_proto(proto))

        # Get all the node names from proto_strings
        node_names = {n.name for n in self._graph_def.node}
        for string in proto_strings:
            if string != "":
                name = self.get_node_name_and_output_index(string)[0]
                if name in node_names:
                    self._required_nodes.add(name)

        # Add required nodes to meta_graph_info
        meta_graph_info.required_nodes.extend(self._required_nodes)

        return meta_graph_info

    def as_light_graph(self):
        # Nodes, inputs, outputs
        nodes = [
            self._tf_node_def_to_lnf(tf_node_def) for tf_node_def in self._graph_def.node
        ]
        input_edges = [
            self._tensor_name_to_edge_info(tensor_name)
            for tensor_name in self._input_tensor_names
        ]
        output_edges = [
            self._tensor_name_to_edge_info(tensor_name)
            for tensor_name in self._output_tensor_names
        ]
        output_node_names = sorted(self._output_node_names)
        meta_graph_info = self._tf_meta_graph_def_to_lgf_meta_graph_info(
            self._meta_graph_def)

        return lgf_graph.LightGraph(nodes,
                                    input_edges=input_edges,
                                    output_edges=output_edges,
                                    output_node_names=output_node_names,
                                    meta_graph_info=meta_graph_info)
