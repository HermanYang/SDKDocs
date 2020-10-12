from lt_sdk.graph.transform_graph.node_transformers.generic_transforms import (
    base_transform,
)
from lt_sdk.proto import lgf_pb2


class ElectronicOpTransform(base_transform.BaseTransform):
    """Contains common functions among electronic ops"""

    def create_simple_node(self,
                           node_name,
                           node_type,
                           inputs,
                           outputs,
                           control_inputs,
                           supported=True):
        new_node = lgf_pb2.LNF()
        new_node.name = node_name
        new_node.supported = supported
        getattr(new_node, node_type).SetInParent()

        # Inputs
        for inp in inputs:
            new_node.inputs.add().CopyFrom(inp)
            new_node.inputs[-1].dtype.CopyFrom(self._get_dtype())

        # Outputs
        for outp in outputs:
            new_node.outputs.add().CopyFrom(outp)
            new_node.outputs[-1].dtype.CopyFrom(self._get_dtype())

        # Control inputs

        new_node.control_inputs.extend(control_inputs)

        return new_node

    def check_unary(self, node, light_graph):
        return (len(node.inputs) == 1 and len(node.outputs) == 1
                and super().can_transform(node,
                                          light_graph))

    def check_single_output(self, node, light_graph):
        return (len(node.outputs) == 1 and super().can_transform(node, light_graph))
