from sdk2.graph.export_graph import graph_exporter_map
from sdk2.graph.import_graph import graph_importer_map
from sdk2.graph.transform_graph import standard_transformations, utils
from sdk2.proto import lgf_pb2


def extract_edge_from_data(calibration_data):
    """Return an EdgeInfo for the input data, in case it isn't specified in the graph."""
    input_edges = []
    for named_tensor in calibration_data.batches[0].inputs:
        e = lgf_pb2.EdgeInfo()
        e.CopyFrom(named_tensor.edge_info)
        if e.shape.batch_dim_indx < 0:
            # NOTE: Assumes dim 0 is batch dim
            e.shape.batch_dim_indx = 0
        e.shape.d[e.shape.batch_dim_indx] = -1

        input_edges.append(e)

    return input_edges


def main(import_graph_path, import_graph_type, export_graph_path, export_graph_type,
         calibration_data, hw_specs, sw_config, sim_params):
    """
    Imports, transforms, and exports the graph stored in the graph_path

    Params:
        import_graph_path: path to the graph to import
        import_graph_type: a graph_types_pb2.GraphType enum specifying the type of
            the imported graph
        export_graph_path: path to export the transformed graph
        export_graph_type: a graph_types_pb2.GraphType enum specifying the type of
            the exported graph
        calibration_data: a inference_pb2.BatchedInferenceInput() protobuf (if None
            generate random data)
        hw_specs: a hw_spec_pb2.HardwareSpecs() protobuf
        sw_config: a sw_config_pb2.SoftwareConfig() protobuf
        sim_params: a sim_params_pb2.SimulationParams() protobuf
    """
    input_edges = extract_edge_from_data(calibration_data) if calibration_data else None

    # Import original graph
    utils.log_message("Importing Graph")
    importer_cls = graph_importer_map.GRAPH_IMPORTER_MAP[import_graph_type]
    importer = importer_cls(import_graph_path, sw_config, input_edges=input_edges)
    original_light_graph = importer.as_light_graph()

    # Standard transformations
    utils.log_message("Standard Transformations")
    transformed_light_graph = standard_transformations.main(original_light_graph,
                                                            calibration_data, hw_specs,
                                                            sw_config, sim_params)

    # Export tranformed graph
    utils.log_message("Exporting Graph")
    exporter_cls = graph_exporter_map.GRAPH_EXPORTER_MAP[export_graph_type]
    exporter = exporter_cls(transformed_light_graph, hw_specs, sw_config, sim_params)
    exporter.export_graph(export_graph_path)
    return transformed_light_graph
