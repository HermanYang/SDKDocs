from sdk2.graph.transform_graph.calibration import \
    convert_to_activation_scale_calibration_graph
from sdk2.graph.transform_graph.graph_transformers import graph_transform
from sdk2.graph.transform_graph.node_transformers.generic_transforms import (
    base_transform, opu_op_transform)
from sdk2.proto import common_pb2, dtypes_pb2, lgf_pb2, ops_pb2


class ConvertToDebugMode(graph_transform.GraphTransform):

    SKIP_NODES = {
        lgf_pb2.LNF.const.DESCRIPTOR.name,
        lgf_pb2.LNF.cast.DESCRIPTOR.name,
    }

    SKIP_OPS = {
        ops_pb2.UNKNOWN,
        ops_pb2.CONST,
        ops_pb2.CAST,
    }

    def __init__(self, sw_config, hist_coll):
        self._hist_coll = hist_coll
        self._sw_config = sw_config
        self._num_bins = 1 << 12
        self._counter = -1
        self._key_map = {}

    @staticmethod
    def file_friendly_name(name):
        return name.replace("/", "-")

    def _get_new_hist_key(self, edge=None, node=None, x=None, y=None, suffix=None):
        # Updates for new key
        self._counter += 1
        hist_key = self._counter

        # Get name for the map
        if edge is not None:
            name = "{}:{}".format(edge.name, edge.port)
        elif node is not None:
            name = node.name
        if x is not None:
            name += "_tile_x_{}".format(x)
        if y is not None:
            name += "_tile_y_{}".format(y)
        if suffix is not None:
            name += "_{}".format(suffix)

        self._key_map[hist_key] = self.file_friendly_name(name)

        return hist_key

    def _get_num_bins(self, dtype):
        # Special cases for quantized types
        if dtype.t == dtypes_pb2.DT_QINT and dtype.p <= 12:
            num_bins = 1 << dtype.p
        elif dtype.t == dtypes_pb2.DT_QINT and dtype.p <= 12:
            num_bins = 1 << (dtype.p + 1)
        else:
            num_bins = self._num_bins

        return num_bins

    def _insert_collect_hist_node(self, edge):
        num_bins = self._get_num_bins(edge.dtype)
        hist_key = self._get_new_hist_key(edge=edge)
        transforms = convert_to_activation_scale_calibration_graph.\
            ConvertToActivationScaleCalibrationGraph.insert_collect_hist_node(
                edge, self._sw_config, hist_key, num_bins, self._hist_coll)

        return transforms

    def _modify_opu_node(self, opu_node):
        # New opu node
        new_opu_node = lgf_pb2.LNF()
        new_opu_node.CopyFrom(opu_node)

        # Get dtypes for histograms
        before_adc_dtype = dtypes_pb2.DType()
        before_adc_dtype.t = dtypes_pb2.DT_FLOAT
        before_adc_dtype.p = 32
        after_adc_dtype = opu_node.outputs[0].dtype

        # Get num bins for histograms
        before_adc_num_bins = self._get_num_bins(before_adc_dtype)
        after_adc_num_bins = self._get_num_bins(after_adc_dtype)

        # Get hist keys from node
        matmul = opu_op_transform.OPUOpTransform.get_matmul_from_opu_node(new_opu_node)
        hist_keys_before_adc = matmul.hist_keys_before_adc
        hist_keys_after_adc = matmul.hist_keys_after_adc

        # Initialize some things
        hist_keys_before_adc.quant_type = common_pb2.QT_PER_TILE
        hist_keys_after_adc.quant_type = common_pb2.QT_PER_TILE

        # Histogram for each tile
        weight_shape = opu_node.inputs[lgf_pb2.MatMulNode.PHASES_INDEX].shape
        for x in range(weight_shape.d[0]):
            for y in range(weight_shape.d[1]):
                # Add keys to node
                hist_keys_before_adc.keys.append(
                    self._get_new_hist_key(node=opu_node, x=x, y=y, suffix="before_adc"))
                hist_keys_after_adc.keys.append(
                    self._get_new_hist_key(node=opu_node, x=x, y=y, suffix="after_adc"))

                # Initialize histograms
                self._hist_coll.initialize_empty_histogram(hist_keys_before_adc.keys[-1],
                                                           before_adc_num_bins)
                self._hist_coll.initialize_empty_histogram(hist_keys_after_adc.keys[-1],
                                                           after_adc_num_bins)

        return base_transform.BaseTransform.create_transform_result(
            to_replace=[new_opu_node])

    def get_transforms(self, light_graph):
        """Returns the transforms to convert the graph to a debugging mode"""
        transforms = []

        for node in light_graph.nodes():
            if node.WhichOneof("node") in self.SKIP_NODES:
                # Skip constant and cast nodes
                continue

            if node.supported:
                # Collect histograms for all outputs of supported nodes
                for e in node.outputs:
                    transforms.append(self._insert_collect_hist_node(e))

                # Collect histograms of tiles inside an opu op
                if node.WhichOneof("node") in self._sw_config.node_types.opu_nodes:
                    transforms.append(self._modify_opu_node(node))

            else:
                # Collect histograms for outputs of unsupported nodes
                assert (node.HasField(lgf_pb2.LNF.original.DESCRIPTOR.name))
                if node.original.op in self.SKIP_OPS:
                    # Skip unknown, constant, and cast ops
                    continue

                for e in node.outputs:
                    transforms.append(self._insert_collect_hist_node(e))

        return transforms

    def get_key_map(self):
        return self._key_map
