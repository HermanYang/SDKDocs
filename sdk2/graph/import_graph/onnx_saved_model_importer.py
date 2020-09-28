import collections
import logging

import onnx
import onnxruntime
from onnx import shape_inference

from sdk2.graph import lgf_graph
from sdk2.graph.import_graph import graph_importer
from sdk2.proto import common_pb2, dtypes_pb2, graph_types_pb2, lgf_pb2, ops_pb2


class ImportONNXModel(graph_importer.ImportGraph):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._input_tensor_names = None
        self._output_tensor_names = None
        self._output_node_names = None

    # onnx assigned value to lgf datatype map
    DATA_TYPE_MAP = {
        0: (dtypes_pb2.DT_INVALID, 0),
        1: (dtypes_pb2.DT_FLOAT, 64),
        11: (dtypes_pb2.DT_FLOAT, 64),
        10: (dtypes_pb2.DT_FLOAT, 16),
        9: (dtypes_pb2.DT_BOOL, 1),
        15: (dtypes_pb2.DT_COMPLEX, 128),
        14: (dtypes_pb2.DT_COMPLEX, 64),
        3: (dtypes_pb2.DT_INT, 8),
        5: (dtypes_pb2.DT_INT, 16),
        6: (dtypes_pb2.DT_INT, 32),
        7: (dtypes_pb2.DT_INT, 64),
        8: (dtypes_pb2.DT_STRING, 0),
        2: (dtypes_pb2.DT_UINT, 8),
        4: (dtypes_pb2.DT_UINT, 16),
        12: (dtypes_pb2.DT_UINT, 32),
        13: (dtypes_pb2.DT_UINT, 64)
    }

    OP_MAP = {
        "Conv": ops_pb2.CONV2D,
        "MatMul": ops_pb2.MATMUL,
        "Add": ops_pb2.ADD,
        "Constant": ops_pb2.CONST,
        "Relu": ops_pb2.RELU,
        "Reshape": ops_pb2.RESHAPE,
        "AveragePool": ops_pb2.AVGPOOL,
        "MaxPool": ops_pb2.MAXPOOL,
        "Identity": ops_pb2.IDENTITY,
        "BatchNormalization": ops_pb2.BATCHNORM,
        "Pad": ops_pb2.PAD,
        "Mean": ops_pb2.MEAN,
        "Softmax": ops_pb2.SOFTMAX,
        "Assign": ops_pb2.ASSIGN,
        "Squeeze": ops_pb2.SQUEEZE,
    }

    def get_name_and_port(self, node_name):
        split_name = node_name.split(":")
        name = split_name[0]
        port = 0
        if len(split_name) > 1:
            if split_name[1] != "":
                try:
                    port = int(split_name[1])
                except ValueError:
                    logging.warning("Port is not a number")
        return "{}:{}".format(name, port)

    def onnx_node_to_lnf(self, onnx_node):
        lnf = lgf_pb2.LNF()
        lnf.name = onnx_node.name
        lnf.supported = False
        lnf.original.SetInParent()
        lnf.original.t = graph_types_pb2.ONNXModel
        lnf.original.op = ImportONNXModel.OP_MAP.get(onnx_node.op_type, ops_pb2.UNKNOWN)
        lnf.original.serialized_node = onnx_node.SerializeToString()

        # Node inputs in edge info format are stored here
        node_input_edges = [
            self.onnx_edge_to_edge_info(self.get_name_and_port(input_edge))
            for input_edge in onnx_node.input
        ]
        for node_input_edge_info in node_input_edges:
            lnf.inputs.add().CopyFrom(node_input_edge_info)

        node_output_edges = [
            self.onnx_edge_to_edge_info(self.get_name_and_port(output_edge))
            for output_edge in onnx_node.input
        ]

        lnf.outputs.extend(node_output_edges)

        return lnf

    def change_onnx_names_to_lnf_standard(self, onnx_model):
        """ Changing all onnx output names to lnf standard &
            assigning node names if node name field is empty """
        ops_set = set()
        for node in onnx_model.graph.node:
            ops_set.add(node.op_type)

        op_set_counter = collections.defaultdict(int)
        onnx_output_name_to_lgf_output_name = {}
        for node in onnx_model.graph.node:
            if node.name == "":
                node.name = node.op_type + str(op_set_counter[node.op_type])
                op_set_counter[node.op_type] += 1

            for i in range(len(node.output)):
                onnx_output_name_to_lgf_output_name[
                    node.output[i]] = node.name + ":" + str(i)
                node.output[i] = node.name + ":" + str(i)

        for node in onnx_model.graph.node:
            if node.input:
                for i in range(len(node.input)):
                    if node.input[i] in onnx_output_name_to_lgf_output_name.keys():
                        node.input[i] = onnx_output_name_to_lgf_output_name[
                            node.input[i]]
        return onnx_model

    def onnx_edge_to_edge_info(self, onnx_edge_name):
        edge_info = lgf_pb2.EdgeInfo()
        name_and_port_str = self.get_name_and_port(onnx_edge_name)
        name, port = name_and_port_str.split(":")
        edge_info.name = name
        edge_info.port = int(port)
        edge_info.dtype.CopyFrom(
            self._tensor_dtypes[self.get_name_and_port(onnx_edge_name)])
        edge_info.shape.CopyFrom(
            self._tensor_shapes[self.get_name_and_port(onnx_edge_name)])
        return edge_info

    def convert_to_lgf_shape(self, shape):
        lgf_tensor_shape = common_pb2.TensorShape()
        lgf_tensor_shape.batch_dim_indx = -1
        if len(shape) == 0:
            return lgf_tensor_shape
        lgf_tensor_shape.d.extend([-1 if d is None else int(d) for d in shape])
        if lgf_tensor_shape.d[0] == -1:
            lgf_tensor_shape.batch_dim_indx = 0
        return lgf_tensor_shape

    def onnx_dtype_to_lgf_dtype(self, mapping_number):
        t, p = ImportONNXModel.DATA_TYPE_MAP.get(mapping_number,
                                                 (dtypes_pb2.DT_INVALID, 0))
        ret = dtypes_pb2.DType()
        ret.t = t
        ret.p = p
        return ret

    def add_shape_dtype_from_given_valueinfo_tensor(self, value):
        shape = [-1]
        name_and_port_str = self.get_name_and_port(value.name)
        if value.type.tensor_type.shape:
            node_shape_info = value.type.tensor_type.shape
            shape = []
            self._tensor_dtypes[name_and_port_str] = self.onnx_dtype_to_lgf_dtype(
                value.type.tensor_type.elem_type)
            for dimension in node_shape_info.dim:
                dim_val = dimension.dim_value
                if dim_val == 0 or dim_val == "?":
                    shape.append(-1)
                else:
                    shape.append(dim_val)
            self._tensor_shapes[name_and_port_str] = self.convert_to_lgf_shape(shape)
        else:
            self._tensor_shapes[name_and_port_str] = self.convert_to_lgf_shape([-1])

    def as_light_graph(self):
        self._max_ports = {}
        self._variable_values = {}
        self._tensor_shapes = {}
        self._tensor_dtypes = {}

        onnx_model = onnx.load(self._graph_path)
        sess = onnxruntime.InferenceSession(onnx_model.SerializeToString())
        onnx_model = self.change_onnx_names_to_lnf_standard(onnx_model)
        # Run shape inference on the graph and map the respective edge name to shape
        inferred_model = shape_inference.infer_shapes(onnx_model)

        # Add graph input's & output's shape and dim to the dictionary
        for value in onnx_model.graph.input:
            self.add_shape_dtype_from_given_valueinfo_tensor(value)

        for value in onnx_model.graph.output:
            self.add_shape_dtype_from_given_valueinfo_tensor(value)

        # Add graph variables' & constant's shape and dim to dict
        for value in inferred_model.graph.value_info:
            self.add_shape_dtype_from_given_valueinfo_tensor(value)

        graph_nodes = [
            self.onnx_node_to_lnf(onnx_node) for onnx_node in onnx_model.graph.node
        ]

        graph_inputs = [
            self.onnx_edge_to_edge_info(onnx_graph_input_edge.name)
            for onnx_graph_input_edge in sess.get_inputs()
        ]
        graph_outputs = [
            self.onnx_edge_to_edge_info(onnx_graph_output_edge.name)
            for onnx_graph_output_edge in sess.get_outputs()
        ]

        graph_output_node_names = [
            output_node.name for output_node in sess.get_outputs()
        ]
        return lgf_graph.LightGraph(graph_nodes,
                                    input_edges=graph_inputs,
                                    output_edges=graph_outputs,
                                    output_node_names=graph_output_node_names)
