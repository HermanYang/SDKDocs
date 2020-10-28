from lt_sdk.graph import lgf_graph
from lt_sdk.graph.import_graph import graph_importer


class ImportLGFProtobuf(graph_importer.ImportGraph):

    def as_light_graph(self):
        lgf_pb = lgf_graph.LightGraph.read_lgf_pb(self._graph_path)
        light_graph = lgf_graph.LightGraph.lgf_pb_to_graph(lgf_pb)
        return light_graph
