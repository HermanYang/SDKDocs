import numpy as np

from sdk2.graph.transform_graph.graph_transformers import apply_node_map
from sdk2.graph.transform_graph.node_transformers.generic_transforms import (
    base_transform, electronic_op_transform, opu_op_transform)
from sdk2.graph.transform_graph.node_transformers.tf_transforms import \
    tf_batch_matmul_transform
from sdk2.proto import calibration_pb2, lgf_pb2, node_filters


class AddActivationScalesNodeTransform(base_transform.BaseTransform):

    def __init__(self, activation_scales_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._activation_scales_data = activation_scales_data

    def _get_activation_scale_info(self, node_name):
        """
        Returns the scale from self._activation_scales_data that matches
        the given node
        """
        for node_scale_pair in self._activation_scales_data.data:
            if (node_name == node_scale_pair.node_info.node_name):
                assert (node_scale_pair.scale_info.scale > 0)
                return node_scale_pair.scale_info

        raise RuntimeError(
            "Could not find activation scale information for the given node {}".format(
                node_name))


class AddOPUActivationScalesNodeTransform(AddActivationScalesNodeTransform):

    def transform(self, opu_node, light_graph):
        node_name = opu_node.name
        if opu_op_transform.OPUOpTransform.is_part_of_batch_matmul(
                opu_node, light_graph):
            # Currently all unstacked matmul nodes from a batch matmul node
            # share the activation scale. The scale is stored with the
            # name of the original batch matmul node.
            node_name = (tf_batch_matmul_transform.TFSavedModelBatchMatMulV2Transform.
                         get_batch_matmul_node_name(node_name))
        activation_scale_info = self._get_activation_scale_info(node_name)

        # Update the quant params node
        quant_params_node = light_graph.get_node_by_name(
            opu_node.inputs[lgf_pb2.MatMulNode.QUANT_PARAMS_INDEX].name)
        new_quant_params = np.array(
            [[activation_scale_info.scale] * self._hw_specs.dimension,
             [activation_scale_info.bias] * self._hw_specs.dimension])

        new_quant_params_node = self.create_const_node(
            new_quant_params, quant_params_node.name, quant_params_node.outputs[0].dtype,
            quant_params_node.const.const_type)

        # Update the opu node
        new_opu_node = lgf_pb2.LNF()
        new_opu_node.CopyFrom(opu_node)
        matmul = opu_op_transform.OPUOpTransform.get_matmul_from_opu_node(new_opu_node)
        matmul.using_quant_bias = (activation_scale_info.bias != 0)
        new_opu_node.inputs[lgf_pb2.MatMulNode.QUANT_PARAMS_INDEX].CopyFrom(
            new_quant_params_node.outputs[0])

        # Update phasify node
        phasify_node = light_graph.get_node_by_name(
            opu_node.inputs[lgf_pb2.MatMulNode.PHASES_INDEX].name)
        new_phasify_node = lgf_pb2.LNF()
        new_phasify_node.CopyFrom(phasify_node)
        new_phasify_node.inputs[lgf_pb2.PhasifyNode.QUANT_PARAMS_INPUT_INDEX].CopyFrom(
            new_quant_params_node.outputs[0])

        return self.create_transform_result(
            to_replace=[new_quant_params_node, new_opu_node, new_phasify_node])


class AddElectronicActivationScalesNodeTransform(
        AddActivationScalesNodeTransform, electronic_op_transform.ElectronicOpTransform):

    def can_transform(self, node, light_graph):
        # Currently only works for unary instructions
        return self.check_unary(node, light_graph)

    def transform(self, node, light_graph):
        activation_scale_info = self._get_activation_scale_info(node.name)

        # Create new node that has a quant scale set
        # Update current node
        new_node = lgf_pb2.LNF()
        new_node.CopyFrom(node)
        node_type = getattr(new_node, new_node.WhichOneof("node"))
        fields = node_type.DESCRIPTOR.fields_by_name
        assert ("quant_scale" in fields)
        assert ("quant_precision" in fields)
        node_type.quant_scale = activation_scale_info.scale
        node_type.quant_precision = self._sw_config.quantized_electronic_op_precision

        return self.create_transform_result(to_replace=[new_node])


class AddActivationScales(apply_node_map.ApplyNodeMap):

    def __init__(self, hw_specs, sw_config, sim_params, activation_scales_data):
        super().__init__(
            hw_specs, sw_config, {
                self.get_opu_node_filter(sw_config):
                    AddOPUActivationScalesNodeTransform(activation_scales_data, hw_specs,
                                                        sw_config, sim_params),
                self.get_quantized_electronic_nodes_filter(sw_config):
                    AddElectronicActivationScalesNodeTransform(
                        activation_scales_data, hw_specs, sw_config, sim_params),
            })

    @staticmethod
    def get_quantized_electronic_nodes_filter(sw_config):
        return node_filters.which_oneof_filter(*(
            sw_config.node_types.quantized_electronic_nodes))

    @staticmethod
    def get_nodes_to_calibrate(light_graph, sw_config):
        opu_filt = AddActivationScales.get_opu_node_filter(sw_config)
        electronic_filt = AddActivationScales.get_quantized_electronic_nodes_filter(
            sw_config)

        nodes_to_calibrate = []
        # Keep track of which nodes have been added. This is necessary to
        # avoid adding the same node multiple times in the presence of
        # batch matmul node.
        added_node_info = set()

        def add_node_info(node_name, precision):
            if node_name not in added_node_info:
                node_info = calibration_pb2.NodeInfo()
                node_info.node_name = node_name
                node_info.quant_precision = precision
                nodes_to_calibrate.append(node_info)
                added_node_info.add(node_name)

        for node in light_graph.nodes():
            if opu_filt.matches(node, light_graph):
                matmul = opu_op_transform.OPUOpTransform.get_matmul_from_opu_node(node)
                if opu_op_transform.OPUOpTransform.is_part_of_batch_matmul(
                        node, light_graph):
                    # Add the original batch matmul node and only add
                    # it at the first unstacked matmul node
                    add_node_info(
                        tf_batch_matmul_transform.TFSavedModelBatchMatMulV2Transform.
                        get_batch_matmul_node_name(node.name), matmul.quant_precision)
                else:
                    add_node_info(node.name, matmul.quant_precision)
            elif electronic_filt.matches(node, light_graph):
                add_node_info(node.name, sw_config.quantized_electronic_op_precision)

        return nodes_to_calibrate
