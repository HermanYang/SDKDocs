from lt_sdk.graph.export_graph import graph_exporter_map


def infer_export_graph_type(graph, config):
    for graph_type, exporter_cls in graph_exporter_map.GRAPH_EXPORTER_MAP.items():
        try:
            exporter = exporter_cls(graph,
                                    config.hw_specs,
                                    config.sw_config,
                                    config.sim_params)

            return graph_type, exporter
        except Exception:
            continue

    raise TypeError("Provided graph is not exportable")


def get_graph_exporter(graph_type):
    return graph_exporter_map.GRAPH_EXPORTER_MAP[graph_type]
