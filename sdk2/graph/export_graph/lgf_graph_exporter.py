from sdk2.graph.export_graph import graph_exporter


class ExportLGFProtobuf(graph_exporter.ExportGraph):
    """Export graph as a LGF Protobuf"""

    def export_graph(self, graph_path):
        self._light_graph.write_lgf_pb(self._light_graph.as_lgf_pb(), graph_path)
