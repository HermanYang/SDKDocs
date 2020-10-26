import importlib

MODULE_PATHS = [
    "lt_sdk.graph.transform_graph.node_transformers.generic_transforms.",
    "lt_sdk.graph.transform_graph.node_transformers.tf_transforms."
]


def get_node_transform(node_transform):
    """
    params:
        node_transform: A sw_config_pb2.NodeTransform
    """
    ret = None
    found = False
    mod, cls_name = node_transform.transform_module_name.split(".")
    for pre in MODULE_PATHS:
        try:
            fullmod = pre + mod
            ret = importlib.import_module(fullmod)
            found = True
            break
        except ModuleNotFoundError:
            found = False

    if not found:
        raise ValueError("Couldn't find {0}".format(mod))
    return getattr(ret, cls_name)
