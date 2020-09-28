from sdk2.graph.transform_graph.node_transformers.generic_transforms import \
    pool_transform
from sdk2.graph.transform_graph.node_transformers.tf_transforms import tf_base_transform
from sdk2.proto import lgf_pb2, ops_pb2


class TFSavedModelPoolTransform(tf_base_transform.TFSavedModelBaseTransform,
                                pool_transform.PoolTransform):

    def transform(self, pool_node, light_graph):
        """
        Converts original node to a supported pool in standard format
        """
        self.check_original_node(pool_node, graph_type=self.GRAPH_TYPE)
        tf_attr = self._get_tf_attr(pool_node)

        # Get the pooling type
        if pool_node.original.op == ops_pb2.MAXPOOL:
            pooling_type = lgf_pb2.PoolNode.MAX_POOL
        elif pool_node.original.op == ops_pb2.AVGPOOL:
            pooling_type = lgf_pb2.PoolNode.AVG_POOL
        else:
            raise NotImplementedError("Unsupported pooling_type")

        # Get kernel size
        kernel_size = list(tf_attr["ksize"].list.i)
        if len(kernel_size) == 1:
            kernel_size += kernel_size

        # Get strides
        strides = list(tf_attr["strides"].list.i)
        if len(strides) == 1:
            strides += strides

        # Padding and data format
        padding = self._get_string(tf_attr, "padding")
        data_format = self._get_string(tf_attr, "data_format")

        return self.do_generic_transform(pool_node.name, pool_node.inputs[0],
                                         pool_node.outputs[0], pool_node.control_inputs,
                                         pooling_type, kernel_size, strides, padding,
                                         data_format)
