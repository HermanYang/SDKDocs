import numpy as np

from sdk2.graph.transform_graph.graph_transformers import apply_node_map
from sdk2.graph.transform_graph.node_transformers.generic_transforms import (
    base_transform, opu_op_transform)
from sdk2.proto import common_pb2, lgf_pb2


class AddADCScalesNodeTransform(base_transform.BaseTransform):

    def __init__(self, adc_scales_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._adc_scales_data = adc_scales_data

    def _get_adc_scales(self, opu_node):
        # Get scales list from the proto
        flat_scales = None
        for node_scale_pair in self._adc_scales_data.data:
            if node_scale_pair.node_info.node_name == opu_node.name:
                assert (len(node_scale_pair.scale_info_list.l) > 0)
                flat_scales = np.array(
                    [s.scale for s in node_scale_pair.scale_info_list.l])

        if flat_scales is None:
            raise ValueError("Could not find adc scale for node {}".format(
                opu_node.name))

        # Get shape of adc scales and original number of colums
        num_x, num_y, _, j = opu_node.inputs[lgf_pb2.MatMulNode.ADC_SCALES_INDEX].shape.d

        # Get shaped adc scales and an array to multiply by dequant scales
        quant_type = self._sw_config.adc_scale_quantization_type
        if quant_type == common_pb2.QT_SINGLE:
            assert (flat_scales.size == 1)
            adc_scales = flat_scales.reshape(1, 1, 1, 1)
        elif quant_type == common_pb2.QT_PER_TILE:
            assert (flat_scales.size == num_x * num_y)
            adc_scales = flat_scales.reshape(num_x, num_y, 1, 1)
        elif quant_type == common_pb2.QT_PER_COL:
            assert (flat_scales.size == num_y * j)
            adc_scales = flat_scales.reshape(1, num_y, 1, j)
        elif quant_type == common_pb2.QT_PER_COL_PER_TILE:
            assert (flat_scales.size == num_x * num_y * j)
            adc_scales = flat_scales.reshape(num_x, num_y, 1, j)

        adc_scales = np.broadcast_to(adc_scales, (num_x, num_y, 1, j))

        return adc_scales

    def transform(self, opu_node, light_graph):
        # New opu node
        new_opu_node = lgf_pb2.LNF()
        new_opu_node.CopyFrom(opu_node)
        matmul = opu_op_transform.OPUOpTransform.get_matmul_from_opu_node(new_opu_node)
        matmul.turn_off_adc = False

        # Get the old adc scales node
        phasify_node = light_graph.get_node_by_name(
            opu_node.inputs[lgf_pb2.MatMulNode.PHASES_INDEX].name)
        adc_scales_node = light_graph.get_node_by_name(
            phasify_node.inputs[lgf_pb2.PhasifyNode.ADC_SCALES_INPUT_INDEX].name)

        # New adc scales node
        new_adc_scales = self._get_adc_scales(opu_node)
        new_adc_scales_node = self.create_const_node(new_adc_scales,
                                                     adc_scales_node.name,
                                                     adc_scales_node.outputs[0].dtype,
                                                     adc_scales_node.const.const_type)

        return self.create_transform_result(
            to_replace=[new_opu_node, new_adc_scales_node])


class AddADCScales(apply_node_map.ApplyNodeMap):

    def __init__(self, hw_specs, sw_config, sim_params, adc_scales_data):
        super().__init__(
            hw_specs, sw_config, {
                self.get_opu_node_filter(sw_config):
                    AddADCScalesNodeTransform(adc_scales_data, hw_specs, sw_config,
                                              sim_params)
            })
