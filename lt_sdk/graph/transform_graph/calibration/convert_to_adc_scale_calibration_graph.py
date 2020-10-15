from lt_sdk.graph.transform_graph.graph_transformers import apply_node_map
from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    base_transform,
    opu_op_transform,
)
from lt_sdk.proto import calibration_pb2, common_pb2, lgf_pb2


class ConvertToADCScaleCalibrationNodeTransform(base_transform.BaseTransform):

    def __init__(self, hist_coll, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hist_coll = hist_coll
        self._num_bins = self._sw_config.adc_scale_num_bins
        self._counter = -1
        self._node_name_to_keys = {}

    def _get_hist_keys(self, opu_node, light_graph):
        # Get shape of adc scales
        num_x, num_y, _, j = opu_node.inputs[lgf_pb2.MatMulNode.ADC_SCALES_INDEX].shape.d

        # Initialize hist_keys
        hist_keys = common_pb2.HistKeys()
        hist_keys.quant_type = self._sw_config.adc_scale_quantization_type

        # Figure out how many keys we need to add
        if hist_keys.quant_type == common_pb2.QT_SINGLE:
            num_keys = 1
        elif hist_keys.quant_type == common_pb2.QT_PER_TILE:
            num_keys = num_x * num_y
        elif hist_keys.quant_type == common_pb2.QT_PER_COL:
            num_keys = num_y * j
        elif hist_keys.quant_type == common_pb2.QT_PER_COL_PER_TILE:
            num_keys = num_x * num_y * j

        # Add keys to hist_keys
        for index in range(num_keys):
            # Updates for a new key
            self._counter += 1
            self._node_name_to_keys.setdefault(opu_node.name, []).append(self._counter)

            # Add the new key to hist_keys
            key = self._node_name_to_keys[opu_node.name][-1]
            hist_keys.keys.append(key)
            self._hist_coll.initialize_empty_histogram(key, self._num_bins)

            # Mark padding histograms
            if opu_op_transform.OPUOpTransform.is_padding(opu_node,
                                                          light_graph,
                                                          self._sw_config,
                                                          index,
                                                          hist_keys.quant_type):
                self._hist_coll.update_histogram_mode(key, calibration_pb2.HM_PADDING)

        return hist_keys

    def node_name_to_keys(self, node_name):
        return self._node_name_to_keys.get(node_name, [])

    def transform(self, opu_node, light_graph):
        # Get hist path and write an empty hist
        hist_keys = self._get_hist_keys(opu_node, light_graph)

        # New opu node
        new_opu_node = lgf_pb2.LNF()
        new_opu_node.CopyFrom(opu_node)
        matmul = opu_op_transform.OPUOpTransform.get_matmul_from_opu_node(new_opu_node)
        matmul.turn_off_adc = True
        matmul.hist_keys_before_adc.CopyFrom(hist_keys)

        return self.create_transform_result(to_replace=[new_opu_node])


class ConvertToADCScaleCalibrationGraph(apply_node_map.ApplyNodeMap):

    def __init__(self, hw_specs, sw_config, sim_params, hist_coll):
        self._node_transform = ConvertToADCScaleCalibrationNodeTransform(
            hist_coll,
            hw_specs,
            sw_config,
            sim_params)
        super().__init__(hw_specs,
                         sw_config,
                         {self.get_opu_node_filter(sw_config): self._node_transform})

    def node_name_to_keys(self, node_name):
        return self._node_transform.node_name_to_keys(node_name)
