import pprint

from lt_sdk.graph.import_graph import graph_importer_map


def infer_import_graph_type(graph_path, config, input_edges=None):
    """

    Args:
        graph_path (str): [description]
        config (light_config.LightConfig): [description]

    Raises:
        TypeError: [description]

    Returns:
        [type]: [description]
    """
    for graph_type, importer_cls in graph_importer_map.GRAPH_IMPORTER_MAP.items():
        try:
            importer = importer_cls(graph_path,
                                    config.sw_config,
                                    input_edges=input_edges)
            _ = importer.as_light_graph()
            return graph_type
        except Exception:
            continue

    raise TypeError(f"Provided graph {graph_path} is not a valid graph type. \
                      The graph must be one of the following types:\n\
                      {pprint.pformat(graph_importer_map.keys())}")


def get_graph_importer(graph_type):
    return graph_importer_map.GRAPH_IMPORTER_MAP[graph_type]
