import logging
import types

import numpy as np

from sdk2.graph import lgf_graph
from sdk2.graph.run_graph import graph_runner
from sdk2.graph.transform_graph import utils
from sdk2.graph.transform_graph.graph_transformers import apply_node_map
from sdk2.graph.transform_graph.node_transformers.generic_transforms import (
    base_transform, opu_op_transform)
from sdk2.proto import lgf_pb2, ops_pb2


class FoldPhasifyConstantsNodeTransform(base_transform.BaseTransform):

    def _get_foldable_phasify_nodes(self, opu_node, light_graph):
        """Returns a tuple in the form of (phasify_node, phasify_subgraph_nodes) where
        phasify_subgraph_nodes constitue the subgraph for folding the phasify_node.

        In case that the phasify_node cannot be folded, (phasify_node, None) will be
        returned.
        """
        phasify_node = light_graph.get_node_by_name(
            opu_node.inputs[lgf_pb2.MatMulNode.PHASES_INDEX].name)

        if opu_op_transform.OPUOpTransform.is_part_of_batch_matmul(
                opu_node, light_graph):
            logging.warning("Can't fold %s because %s is part of a batch matmul node.",
                            phasify_node.name, opu_node.name)
            return phasify_node, None

        if not light_graph.is_constant_node(phasify_node):
            logging.warning("Can't fold %s because it is not a constant node.",
                            phasify_node.name)
            return phasify_node, None

        phasify_subgraph_nodes = [n for n in light_graph.bfs(phasify_node)]
        subgraph_node_names = {n.name for n in phasify_subgraph_nodes}

        # If there is a node asking for an input that is not in
        # the subgraph, then we cannot fold the subgraph
        for node in phasify_subgraph_nodes:
            for inp_edge in node.inputs:
                if inp_edge.name not in subgraph_node_names:
                    logging.warning("Can't fold %s because %s not in subgraph.",
                                    node.name, inp_edge.name)
                    return phasify_node, None

        # More requirements:
        # There cannot be any variables in the subgraph
        for node in phasify_subgraph_nodes:
            if node.HasField(lgf_pb2.LNF.variable.DESCRIPTOR.name):
                logging.warning("Can't fold %s because node is a variable.", node.name)
                return phasify_node, None
            if node.HasField(lgf_pb2.LNF.original.DESCRIPTOR.name):
                if node.original.op == ops_pb2.VARIABLE:
                    logging.warning(
                        "Can't fold %s because node is an unsupported variable.",
                        node.name)
                    return phasify_node, None

        return phasify_node, phasify_subgraph_nodes

    def _get_weight_rows(self, light_graph, phasify_node):
        weights_shape = opu_op_transform.OPUOpTransform.get_2d_weights_shape(
            phasify_node.inputs[lgf_pb2.PhasifyNode.WEIGHTS_INPUT_INDEX],
            phasify_node.phasify.transpose)
        return weights_shape[0]

    @staticmethod
    def _get_top_and_bottom_zero_rows(array):
        assert (len(array.shape) == 2)
        k = array.shape[0]
        top_zero_rows = 0
        bottom_zero_rows = 0

        for i in range(k):
            if np.all(array[i, :] == 0):
                top_zero_rows += 1
            else:
                break

        for i in range(k - 1, -1, -1):
            if np.all(array[i, :] == 0):
                bottom_zero_rows += 1
            else:
                break

        return top_zero_rows, bottom_zero_rows

    def _add_block_sparsity(self, phases_node, phases_array):
        num_x, num_y, num_dps, k, _ = phases_array.shape
        weight_rows = phases_node.const.weight_rows

        # These must be mutable to be modified in the helper function
        sparse_rows_per_tile = types.SimpleNamespace(val=None)
        sparsity_type = []

        def add_larger(top_zero_rows, bottom_zero_rows):
            if top_zero_rows > bottom_zero_rows:
                sparse_rows_per_tile.val = top_zero_rows
                sparsity_type.append(lgf_pb2.SPARSE_TOP)
            else:
                sparse_rows_per_tile.val = bottom_zero_rows
                sparsity_type.append(lgf_pb2.SPARSE_BOTTOM)

        for x in range(num_x):
            for y in range(num_y):
                for d in range(num_dps):
                    block = phases_array[x, y, d]

                    # Remove padding from last x block
                    block_rows = min(k, weight_rows - x * k)
                    block = block[:block_rows]

                    # No block sparsity in last x block for certain cases
                    if block_rows < k / 2:
                        sparsity_type.append(lgf_pb2.SPARSE_NONE)
                        continue

                    # Get the number of zero rows on the top and bottom
                    top_zero_rows, bottom_zero_rows = self._get_top_and_bottom_zero_rows(
                        block)

                    # Matrix is not in correct sparse format
                    if top_zero_rows == 0 and bottom_zero_rows == 0:
                        return

                    if sparse_rows_per_tile.val is not None:
                        # If num_sparse_rows is set, we can decrease it if necessary
                        if (top_zero_rows <= sparse_rows_per_tile.val
                                and bottom_zero_rows <= sparse_rows_per_tile.val):
                            add_larger(top_zero_rows, bottom_zero_rows)
                        elif (bottom_zero_rows > sparse_rows_per_tile.val):
                            sparsity_type.append(lgf_pb2.SPARSE_BOTTOM)
                        else:
                            assert (top_zero_rows > sparse_rows_per_tile.val)
                            sparsity_type.append(lgf_pb2.SPARSE_TOP)
                    else:
                        # Initialize num_sparse_rows with larger value
                        add_larger(top_zero_rows, bottom_zero_rows)

        if sparse_rows_per_tile.val is None or sparse_rows_per_tile.val == 0:
            return

        # Extra checks for valid sparse layout
        non_sparse_weight_rows = -1
        for y in range(num_y):
            for d in range(num_dps):
                row_count = 0
                for x in range(num_x):
                    i = np.ravel_multi_index((x, y, d), (num_x, num_y, num_dps))
                    sparsity = sparsity_type[i]
                    block_rows = min(k, weight_rows - x * k)

                    if sparsity == lgf_pb2.SPARSE_NONE:
                        row_count += block_rows
                    elif (sparsity == lgf_pb2.SPARSE_BOTTOM
                          or sparsity == lgf_pb2.SPARSE_TOP):
                        row_count += block_rows - sparse_rows_per_tile.val
                    else:
                        raise ValueError("Invalid sparsity: {}".format(sparsity))

                if y == 0 and d == 0:
                    non_sparse_weight_rows = row_count
                elif (non_sparse_weight_rows != row_count):
                    return

        # If we got here, then the sparsity format is valid
        assert (len(sparsity_type) == num_x * num_y * num_dps)
        phases_node.const.non_sparse_weight_rows = non_sparse_weight_rows
        phases_node.const.sparse_rows_per_tile = sparse_rows_per_tile.val
        phases_node.const.sparsity_type.extend(sparsity_type)

    def _get_phases_and_dequant_scales(self, new_opu_node, light_graph, phasify_node,
                                       phasify_subgraph_nodes, transform_result):
        # Create a subgraph
        subgraph = lgf_graph.LightGraph(phasify_subgraph_nodes,
                                        output_edges=phasify_node.outputs)

        # Run the graph
        subgraph_inputs = utils.create_inference_inputs([], [])
        runner = graph_runner.GraphRunner(subgraph, self._hw_specs, self._sw_config,
                                          self._sim_params)
        out_inf = runner.run_single_batch(subgraph_inputs)

        # Get numpy arrays
        phasify_output_arrays = [
            utils.tensor_pb_to_array(named_tensor.data, np.float32)
            for named_tensor in out_inf.results
        ]

        # Get the adc scales node
        adc_scales_node = light_graph.get_node_by_name(
            phasify_node.inputs[lgf_pb2.PhasifyNode.ADC_SCALES_INPUT_INDEX].name)

        # Create constant nodes
        phases_node = self.create_const_node(
            phasify_output_arrays[lgf_pb2.PhasifyNode.PHASES_OUTPUT_INDEX],
            new_opu_node.name + "_phases",
            new_opu_node.inputs[lgf_pb2.MatMulNode.PHASES_INDEX].dtype,
            lgf_pb2.ConstNode.WEIGHTS)
        phases_node.const.weight_rows = self._get_weight_rows(light_graph, phasify_node)
        if not self._sw_config.disable_block_sparsity:
            self._add_block_sparsity(
                phases_node,
                phasify_output_arrays[lgf_pb2.PhasifyNode.PHASES_OUTPUT_INDEX])
        dequant_scales_node = self.create_const_node(
            phasify_output_arrays[lgf_pb2.PhasifyNode.DEQUANT_SCALES_OUTPUT_INDEX],
            new_opu_node.name + "_dequant_scales",
            new_opu_node.inputs[lgf_pb2.MatMulNode.DEQUANT_SCALES_INDEX].dtype,
            lgf_pb2.ConstNode.DEQUANT_SCALE)
        new_adc_scales_node = self.create_const_node(
            phasify_output_arrays[lgf_pb2.PhasifyNode.ADC_SCALES_OUTPUT_INDEX],
            adc_scales_node.name,
            new_opu_node.inputs[lgf_pb2.MatMulNode.ADC_SCALES_INDEX].dtype,
            adc_scales_node.const.const_type)

        # Update the new opu node inputs
        new_opu_node.inputs[lgf_pb2.MatMulNode.PHASES_INDEX].CopyFrom(
            phases_node.outputs[0])
        new_opu_node.inputs[lgf_pb2.MatMulNode.DEQUANT_SCALES_INDEX].CopyFrom(
            dequant_scales_node.outputs[0])
        new_opu_node.inputs[lgf_pb2.MatMulNode.ADC_SCALES_INDEX].CopyFrom(
            new_adc_scales_node.outputs[0])

        transform_result.CopyFrom(
            self.create_transform_result(to_add=[phases_node, dequant_scales_node],
                                         to_replace=[new_adc_scales_node]))

    def _get_dequant_bias(self, new_opu_node, light_graph, transform_result):
        # Create a dequant bias with all 0's
        _, num_y, _, j = new_opu_node.inputs[
            lgf_pb2.MatMulNode.DEQUANT_SCALES_INDEX].shape.d
        zero_dequant_bias_node = self.create_const_node(
            np.zeros([1, num_y, 1, j]), new_opu_node.name + "_dequant_bias",
            self._sw_config.float_type, lgf_pb2.ConstNode.DEQUANT_BIAS)

        # Add dequant bias to the new opu node
        new_opu_node.inputs.add().CopyFrom(zero_dequant_bias_node.outputs[0])

        # Get constant nodes from the transform result
        const_nodes = [
            t.node
            for t in list(transform_result.to_add) + list(transform_result.to_replace)
            if t.node.HasField(lgf_pb2.LNF.const.DESCRIPTOR.name)
        ]
        const_nodes.append(
            light_graph.get_node_by_name(
                new_opu_node.inputs[lgf_pb2.MatMulNode.QUANT_PARAMS_INDEX].name))

        # Create a subgraph to run the new opu node
        subgraph = lgf_graph.LightGraph(
            const_nodes + [new_opu_node, zero_dequant_bias_node],
            input_edges=[new_opu_node.inputs[lgf_pb2.MatMulNode.INPUT_INDEX]],
            output_edges=[new_opu_node.outputs[0]])

        # Create zero inputs
        input_edge = subgraph.input_edges()[0]
        array = np.zeros([
            d if d != -1 else self._sim_params.compiled_batch_size
            for d in input_edge.shape.d
        ])
        zero_inputs = utils.create_inference_inputs([input_edge], [array])

        # Run zeros through the subgraph
        runner = graph_runner.GraphRunner(subgraph, self._hw_specs, self._sw_config,
                                          self._sim_params)
        out_inf = runner.run_single_batch(zero_inputs)

        # New bias is chosen so the outputs of a zero input will be exactly zero
        dequant_bias = -1 * utils.tensor_pb_to_array(out_inf.results[0].data, np.float32)

        # Convert to a [1, last_dim] vector
        dequant_bias = np.reshape(dequant_bias, [-1, dequant_bias.shape[-1]])
        dequant_bias = dequant_bias[0:1, :]

        # Pad and reshape so dequant_bias is [1, num_y, 1, j]
        pad = [[0, 0], [0, num_y * j - dequant_bias.shape[1]]]
        dequant_bias = np.pad(dequant_bias, pad, "constant", constant_values=0)
        dequant_bias = np.split(dequant_bias, num_y, axis=1)
        dequant_bias = np.stack(dequant_bias, axis=0)
        dequant_bias = np.reshape(dequant_bias, [1, num_y, 1, j])

        # Create a dequant bias node and add to the transform result
        dequant_bias_node = self.create_const_node(
            dequant_bias, zero_dequant_bias_node.name,
            zero_dequant_bias_node.outputs[0].dtype,
            zero_dequant_bias_node.const.const_type)

        transform_result.to_add.add().node.CopyFrom(dequant_bias_node)

    def transform(self, opu_node, light_graph):
        transform_result = self.create_transform_result()

        phasify_node, phasify_subgraph_nodes = self._get_foldable_phasify_nodes(
            opu_node, light_graph)

        if phasify_subgraph_nodes is None:
            return transform_result

        # Create a new opu node
        new_opu_node = lgf_pb2.LNF()
        new_opu_node.CopyFrom(opu_node)
        matmul = opu_op_transform.OPUOpTransform.get_matmul_from_opu_node(new_opu_node)
        matmul.phasify_is_folded = True

        # Get phases and dequant scales
        self._get_phases_and_dequant_scales(new_opu_node, light_graph, phasify_node,
                                            phasify_subgraph_nodes, transform_result)

        # Compute the dequant bias if necessary
        if matmul.using_quant_bias:
            self._get_dequant_bias(new_opu_node, light_graph, transform_result)

        transform_result.to_replace.add().node.CopyFrom(new_opu_node)
        return transform_result


class FoldPhasifyConstants(apply_node_map.ApplyNodeMap):

    def __init__(self, hw_specs, sw_config, sim_params):
        super().__init__(
            hw_specs, sw_config, {
                self.get_opu_node_filter(sw_config):
                    FoldPhasifyConstantsNodeTransform(hw_specs, sw_config, sim_params)
            })
