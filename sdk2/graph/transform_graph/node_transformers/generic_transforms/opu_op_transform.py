import numpy as np

from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    base_transform
from sdk2.proto import common_pb2, lgf_pb2
from sdk2.proto.configs import generate_hw_specs


class OPUOpTransform(base_transform.BaseTransform):

    # Dequant bias is optional input, don't include by defualt
    NUM_INPUTS = len(lgf_pb2.MatMulNode.index.DESCRIPTOR.values) - 1
    PHASIFY_NUM_INPUTS = len(lgf_pb2.PhasifyNode.input_index.DESCRIPTOR.values)
    PHASIFY_NUM_OUTPUTS = len(lgf_pb2.PhasifyNode.output_index.DESCRIPTOR.values)

    @property
    def _phase_precision(self):
        """Returns the effective phase precision for the op"""
        phase_precision = self._hw_specs.phase_precision
        if self._hw_specs.time_multiplex_weights:
            phase_precision *= 2
        return phase_precision

    @property
    def _phase_dtype(self):
        return self._get_qint_dtype(self._phase_precision, self._hw_specs.signed_phases)

    @property
    def _quantize_precision(self):
        input_precision = self._hw_specs.input_precision
        if self._hw_specs.time_multiplex_inputs:
            input_precision *= 2
        return input_precision

    @property
    def _matmul_dequantize_method(self):
        return lgf_pb2.DQ_STANDARD

    def num_columns(self):
        return generate_hw_specs.get_num_columns(self._hw_specs)

    def _set_default_attributes(self, matmul):
        matmul.turn_off_adc = False
        matmul.hist_keys_before_adc.SetInParent()
        matmul.hist_keys_after_adc.SetInParent()
        matmul.quant_precision = self._quantize_precision
        matmul.using_quant_bias = False
        matmul.phasify_is_folded = False
        matmul.dequant_method = self._matmul_dequantize_method

    @staticmethod
    def get_2d_weights_shape(weights_edge, transpose_weights):
        # Use a copy so we do not alter the original shape
        weights_shape = weights_edge.shape.d[:]
        if transpose_weights:
            weights_shape[-2:] = weights_shape[-2:][::-1]

        return [np.prod(weights_shape[:-1]), weights_shape[-1]]

    @staticmethod
    def get_tiled_shape(weights_edge, transpose_weights, hw_specs):
        weights_shape = OPUOpTransform.get_2d_weights_shape(weights_edge,
                                                            transpose_weights)
        k = hw_specs.dimension
        j = min(generate_hw_specs.get_num_columns(hw_specs),
                k * np.ceil(weights_shape[1] / k).astype(int))
        num_x = np.ceil(weights_shape[0] / k).astype(int)
        num_y = np.ceil(weights_shape[1] / j).astype(int)

        assert (j % k == 0)
        return num_x, num_y, k, j

    def _create_phasify_node(self, opu_node, weights_edge, transpose_weights=False):
        # Get shape variables from the weights edge
        num_x, num_y, k, j = self.get_tiled_shape(weights_edge, transpose_weights,
                                                  self._hw_specs)

        # Create a phasify node
        phasify_node = lgf_pb2.LNF()
        phasify_node.name = opu_node.name + "_phasify"
        phasify_node.supported = True
        phasify_node.phasify.SetInParent()
        phasify_node.phasify.transpose = transpose_weights

        # Initialize inputs
        for _ in range(self.PHASIFY_NUM_INPUTS):
            phasify_node.inputs.add()

        # Quant params [quant_scale, quant_bias]
        quant_params = np.array([1, 0])
        quant_params_node = self.create_const_node(quant_params,
                                                   opu_node.name + "_quant_params",
                                                   self._sw_config.float_type,
                                                   lgf_pb2.ConstNode.GRAPH_CONST)
        phasify_node.inputs[lgf_pb2.PhasifyNode.QUANT_PARAMS_INPUT_INDEX].CopyFrom(
            quant_params_node.outputs[0])

        # Weights
        phasify_node.inputs[lgf_pb2.PhasifyNode.WEIGHTS_INPUT_INDEX].CopyFrom(
            weights_edge)
        phasify_node.inputs[lgf_pb2.PhasifyNode.WEIGHTS_INPUT_INDEX].dtype.CopyFrom(
            self._sw_config.float_type)

        # ADC scales
        adc_scales = np.ones(shape=[num_x, num_y, 1, j])
        adc_scales_node = self.create_const_node(adc_scales,
                                                 opu_node.name + "_adc_scales",
                                                 self._sw_config.float_type,
                                                 lgf_pb2.ConstNode.ADC_SCALE)
        phasify_node.inputs[lgf_pb2.PhasifyNode.ADC_SCALES_INPUT_INDEX].CopyFrom(
            adc_scales_node.outputs[0])

        # Initialize outputs
        for _ in range(self.PHASIFY_NUM_OUTPUTS):
            phasify_node.outputs.add()

        # Phases
        phases_edge = phasify_node.outputs[lgf_pb2.PhasifyNode.PHASES_OUTPUT_INDEX]
        phases_edge.name = phasify_node.name
        phases_edge.port = lgf_pb2.PhasifyNode.PHASES_OUTPUT_INDEX
        phases_edge.dtype.CopyFrom(self._phase_dtype)
        phases_edge.shape.d.extend([num_x, num_y, j // k, k, k])
        phases_edge.shape.batch_dim_indx = -1

        # Dequant scales
        dequant_scales_edge = phasify_node.outputs[
            lgf_pb2.PhasifyNode.DEQUANT_SCALES_OUTPUT_INDEX]
        dequant_scales_edge.name = phasify_node.name
        dequant_scales_edge.port = lgf_pb2.PhasifyNode.DEQUANT_SCALES_OUTPUT_INDEX
        dequant_scales_edge.dtype.CopyFrom(self._sw_config.float_type)
        dequant_scales_edge.shape.d.extend([num_x, num_y, 1, j])
        dequant_scales_edge.shape.batch_dim_indx = -1

        # ADC scales
        adc_scales_edge = phasify_node.outputs[
            lgf_pb2.PhasifyNode.ADC_SCALES_OUTPUT_INDEX]
        adc_scales_edge.CopyFrom(adc_scales_node.outputs[0])
        adc_scales_edge.name = phasify_node.name
        adc_scales_edge.port = lgf_pb2.PhasifyNode.ADC_SCALES_OUTPUT_INDEX

        # Initialize inputs of the opu node
        for _ in range(self.NUM_INPUTS):
            opu_node.inputs.add()

        # Add input edges from phasify
        opu_node.inputs[lgf_pb2.MatMulNode.QUANT_PARAMS_INDEX].CopyFrom(
            phasify_node.inputs[lgf_pb2.PhasifyNode.QUANT_PARAMS_INPUT_INDEX])
        opu_node.inputs[lgf_pb2.MatMulNode.PHASES_INDEX].CopyFrom(
            phasify_node.outputs[lgf_pb2.PhasifyNode.PHASES_OUTPUT_INDEX])
        opu_node.inputs[lgf_pb2.MatMulNode.DEQUANT_SCALES_INDEX].CopyFrom(
            phasify_node.outputs[lgf_pb2.PhasifyNode.DEQUANT_SCALES_OUTPUT_INDEX])
        opu_node.inputs[lgf_pb2.MatMulNode.ADC_SCALES_INDEX].CopyFrom(
            phasify_node.outputs[lgf_pb2.PhasifyNode.ADC_SCALES_OUTPUT_INDEX])

        # Default attributes
        self._set_default_attributes(self.get_matmul_from_opu_node(opu_node))

        return [phasify_node, quant_params_node, adc_scales_node]

    @staticmethod
    def get_matmul_from_opu_node(opu_node):
        if opu_node.HasField(lgf_pb2.LNF.matmul.DESCRIPTOR.name):
            return opu_node.matmul
        elif opu_node.HasField(lgf_pb2.LNF.conv2d.DESCRIPTOR.name):
            return opu_node.conv2d.matmul
        elif opu_node.HasField(
                lgf_pb2.LNF.block_diagonal_depthwise_conv2d.DESCRIPTOR.name):
            return opu_node.block_diagonal_depthwise_conv2d.conv2d.matmul
        elif opu_node.HasField(lgf_pb2.LNF.distributed_depthwise_conv2d.DESCRIPTOR.name):
            return opu_node.distributed_depthwise_conv2d.conv2d.matmul
        else:
            raise ValueError("Invalid opu node")

    @staticmethod
    def is_padding(opu_node, light_graph, sw_config, index, quant_type):
        """
        Params:
            opu_node: an opu_node with hist keys
            light_graph: light graph that includes the opu_node
            sw_config: a sw_config_pb2.SoftwareConfig() protobuf
            index: index for the hist key to check
            quant_type: quant type for the histograms

        Returns:
            is_padding: let hist_keys be the common_pb2.HistKeys() object in the given
                node. Returns True if hist_keys.keys[index] corresponds to a padding
                column
        """
        assert (opu_node.WhichOneof("node") in sw_config.node_types.opu_nodes)
        if quant_type in [common_pb2.QT_SINGLE, common_pb2.QT_PER_TILE]:
            return False

        # Get some tile information from the opu node
        phasify_node = light_graph.get_node_by_name(
            opu_node.inputs[lgf_pb2.MatMulNode.PHASES_INDEX].name)
        phase_shape = phasify_node.outputs[
            lgf_pb2.PhasifyNode.PHASES_OUTPUT_INDEX].shape.d
        num_x, num_y, num_dps, _, k = phase_shape
        j = num_dps * k
        tile_x, tile_y, col = np.unravel_index(index, (num_x, num_y, j))

        if (opu_node.WhichOneof("node") ==
                lgf_pb2.LNF.block_diagonal_depthwise_conv2d.DESCRIPTOR.name):
            # Need input channels and channel multiplier
            depthwise_reshape_node = light_graph.get_node_by_name(
                phasify_node.inputs[lgf_pb2.PhasifyNode.WEIGHTS_INPUT_INDEX].name)
            _, _, in_channels, channel_multiplier = depthwise_reshape_node.inputs[
                0].shape.d

            # Get the start and end channel in the current tile
            num_tiles = num_x * num_y
            matrices_per_tile = np.ceil(in_channels / num_tiles).astype(int)
            start_channel = matrices_per_tile * tile_y
            end_channel = min(in_channels, start_channel + matrices_per_tile)

            return col >= (end_channel - start_channel) * channel_multiplier
        else:
            # Matmul or conv2d
            weight_shape = phasify_node.inputs[
                lgf_pb2.PhasifyNode.WEIGHTS_INPUT_INDEX].shape.d

            return ((tile_y * j) + col) >= weight_shape[-1]

    @staticmethod
    def is_part_of_batch_matmul(opu_node, light_graph):
        """
        Check whether a node is part of the result of a batch matmul transformation.
        """
        return opu_node.HasField(
            lgf_pb2.LNF.matmul.DESCRIPTOR.name) and opu_node.matmul.from_batch_matmul
