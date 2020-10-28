import numpy as np

from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    base_transform,
    identity_transform,
    matmul_transform,
    reshape_transform,
    stack_transform,
    unstack_transform,
)
from lt_sdk.graph.transform_graph.node_transformers.tf_transforms import (
    tf_base_transform,
)
from lt_sdk.proto import lgf_pb2


class TFSavedModelBatchMatMulV2Transform(base_transform.BaseTransform,
                                         tf_base_transform.TFSavedModelBaseTransform):

    def _unstack_edges(self, edge, num):
        """
        Unstack input edges of a BatchMatMulV2 node to a list of 2D edges which can
        then be fed into regular matmul nodes.

        A reshape node is added if the original input edge has a rank larger than 3.
        """
        common_args = self._common_args()
        new_nodes = []

        batch_prod = 1
        for i, v in enumerate(edge.shape.d[:-2]):
            if v != -1:
                batch_prod *= v
            else:
                assert i == edge.shape.batch_dim_indx

        do_reshape = len(edge.shape.d) > 3
        if do_reshape:
            # Create a reshape node to flatten all but the lowest two dimensions.
            reshape_edge = lgf_pb2.EdgeInfo()
            reshape_edge.name = edge.name + "_reshape"
            reshape_edge.port = 0
            reshape_edge.dtype.CopyFrom(edge.dtype)
            reshape_edge.shape.d[:] = edge.shape.d[-3:]
            if edge.shape.batch_dim_indx == -1:
                reshape_edge.shape.d[0] = batch_prod
                reshape_edge.shape.batch_dim_indx = -1
            elif edge.shape.batch_dim_indx < len(edge.shape.d) - 2:
                reshape_edge.shape.d[0] = -1
                reshape_edge.shape.batch_dim_indx = 0
                reshape_edge.shape.batch_dilation_factor = (
                    edge.shape.batch_dilation_factor)
                reshape_edge.shape.batch_dilation_factor *= batch_prod
            else:
                reshape_edge.shape.d[0] = batch_prod
                reshape_edge.shape.batch_dim_indx = edge.shape.batch_dim_indx - (
                    len(edge.shape.d) - 3)
                reshape_edge.shape.batch_dilation_factor = (
                    edge.shape.batch_dilation_factor)

            reshape_node = reshape_transform.ReshapeTransform(
                *common_args).create_supported_nodes(reshape_edge.name,
                                                     edge,
                                                     reshape_edge,
                                                     [])[0]

            new_nodes.append(reshape_node)
        else:
            reshape_edge = edge

        unstacked_edges = []
        for i in range(num):
            new_edge = lgf_pb2.EdgeInfo()
            new_edge.name = edge.name + "_unstack"
            new_edge.port = i
            new_edge.dtype.CopyFrom(edge.dtype)
            new_edge.shape.d[:] = reshape_edge.shape.d[-2:]
            if reshape_edge.shape.batch_dim_indx >= 1:
                new_edge.shape.batch_dim_indx = reshape_edge.shape.batch_dim_indx - 1
                new_edge.shape.batch_dilation_factor = (
                    reshape_edge.shape.batch_dilation_factor)
            else:
                new_edge.shape.batch_dim_indx = -1
            unstacked_edges.append(new_edge)

        unstack_node = unstack_transform.UnstackTransform(
            *common_args).create_supported_nodes(unstacked_edges[0].name,
                                                 reshape_edge,
                                                 unstacked_edges,
                                                 [],
                                                 0)[0]
        new_nodes.append(unstack_node)

        return new_nodes

    def _stack_edges(self, edges, out_edge):
        """
        Stack output edges of a group of matmul nodes and reshape the result to match the
        given out_edge.
        """
        stack_edge = lgf_pb2.EdgeInfo()
        stack_edge.name = out_edge.name + "_stack"
        stack_edge.port = 0
        stack_edge.dtype.CopyFrom(edges[0].dtype)
        stack_edge.shape.d.append(len(edges))
        stack_edge.shape.d.extend(edges[0].shape.d)
        if edges[0].shape.batch_dim_indx == -1:
            stack_edge.shape.batch_dim_indx = -1
        else:
            stack_edge.shape.batch_dim_indx = edges[0].shape.batch_dim_indx + 1
            stack_edge.shape.batch_dilation_factor = edges[0].shape.batch_dilation_factor

        common_args = self._common_args()
        stack_node = stack_transform.StackTransform(*common_args).create_supported_nodes(
            stack_edge.name,
            edges,
            stack_edge,
            [],
            0)[0]

        if len(out_edge.shape.d) > 3:
            out_node = reshape_transform.ReshapeTransform(
                *common_args).create_supported_nodes(out_edge.name,
                                                     stack_edge,
                                                     out_edge,
                                                     [])[0]
        else:
            out_node = identity_transform.IdentityTransform(
                *common_args).create_supported_nodes(out_edge.name,
                                                     stack_edge,
                                                     out_edge,
                                                     [])[0]

        return [out_node, stack_node]

    def create_supported_nodes(self,
                               batch_matmul_name,
                               input_edge,
                               weight_edge,
                               output_edge,
                               control_inputs,
                               transpose_inputs,
                               transpose_weights,
                               num):
        input_trans = self._unstack_edges(input_edge, num)
        weight_trans = self._unstack_edges(weight_edge, num)

        common_args = self._common_args()
        matmul_obj = matmul_transform.MatMulTransform(*common_args)

        new_matmul_nodes = []
        new_matmul_edges = []
        for i, (new_input, new_weight) in enumerate(
                zip(input_trans[-1].outputs, weight_trans[-1].outputs)):
            matmul_edge = lgf_pb2.EdgeInfo()
            matmul_edge.name = self._get_unstacked_matmul_node_name(batch_matmul_name, i)
            matmul_edge.dtype.CopyFrom(output_edge.dtype)

            # Regular matmul's input edge should not have its batch dim
            # equal to the contracted dim
            if new_input.shape.batch_dim_indx != 1:
                matmul_edge.shape.batch_dim_indx = new_input.shape.batch_dim_indx
                matmul_edge.shape.batch_dilation_factor = (
                    new_input.shape.batch_dilation_factor)
            else:
                matmul_edge.shape.batch_dim_indx = -1

            matmul_edge.shape.d.append(
                new_input.shape.d[1] if transpose_inputs else new_input.shape.d[0])
            matmul_edge.shape.d.append(
                new_weight.shape.d[0] if transpose_weights else new_weight.shape.d[1])
            matmul_node = matmul_obj.create_supported_nodes(matmul_edge.name,
                                                            new_input,
                                                            new_weight,
                                                            matmul_edge,
                                                            control_inputs,
                                                            transpose_inputs,
                                                            transpose_weights)
            matmul_node[0].matmul.from_batch_matmul = True
            new_matmul_nodes.extend(matmul_node)
            new_matmul_edges.append(matmul_edge)

        all_nodes = self._stack_edges(new_matmul_edges, output_edge)
        all_nodes += input_trans + weight_trans + new_matmul_nodes

        return all_nodes

    def transform(self, batch_matmul_node, light_graph):
        """
        Converts original node to a supported matmul in standard format.
        """
        self.check_original_node(batch_matmul_node, graph_type=self.GRAPH_TYPE)
        tf_attr = self._get_tf_attr(batch_matmul_node)

        # Get input and weight nodes
        input_node, _ = self.find_input_and_weight_nodes(batch_matmul_node, light_graph)
        input_index = self._get_input_index(input_node, batch_matmul_node)

        # Get transpose attributes
        transpose_map = {0: "adj_x", 1: "adj_y"}
        transpose_inputs = tf_attr[transpose_map[input_index]].b
        transpose_weights = tf_attr[transpose_map[1 - input_index]].b

        return self.do_generic_transform(
            batch_matmul_node.name,
            batch_matmul_node.inputs[input_index],
            batch_matmul_node.inputs[1 - input_index],
            batch_matmul_node.outputs[0],
            batch_matmul_node.control_inputs,
            transpose_inputs=transpose_inputs,
            transpose_weights=transpose_weights,
            num=self._infer_batch_size(batch_matmul_node.inputs[input_index].shape))

    def _infer_batch_size(self, shape):
        """
        Infer the number of 2D MatMuls contained in the BatchMatMulV2 node from the
        shape of its input edges.
        """
        batch_shape = shape.d[:-2]
        if 0 <= shape.batch_dim_indx < len(batch_shape):
            batch_shape[shape.batch_dim_indx] = self._sim_params.compiled_batch_size
            batch_shape[shape.batch_dim_indx] *= (shape.batch_dilation_factor if
                                                  shape.batch_dilation_factor > 0 else 1)

        return np.prod(batch_shape)

    def can_transform(self, node, light_graph):
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            return False

        edge1 = node.inputs[0]
        edge2 = node.inputs[1]

        if len(edge1.shape.d) != len(edge2.shape.d) or len(edge1.shape.d) <= 2:
            return False

        can_transform = True
        for d1, d2 in zip(edge1.shape.d[:-2], edge2.shape.d[:-2]):
            if (d1 != d2) and (-1 not in (d1, d2)):
                can_transform = False

        return can_transform

    @staticmethod
    def _get_unstacked_matmul_node_name(batch_matmul_name, index):
        return batch_matmul_name + "_BM" + str(index)

    @staticmethod
    def get_batch_matmul_node_name(unstacked_matmul_name):
        return unstacked_matmul_name[:unstacked_matmul_name.rindex("_")]
