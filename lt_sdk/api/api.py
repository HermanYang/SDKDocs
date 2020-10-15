from lt_sdk.funcsim import funcsim
from lt_sdk.graph.export_graph import graph_export
from lt_sdk.graph.import_graph import graph_import
from lt_sdk.graph.transform_graph import standard_transformations as transforms
from lt_sdk.perfsim import perfsim

__all__ = [
    "import_graph",
    "transform_graph",
    "export_graph",
    "run_functional_simulation",
    "run_performance_simulation",
    "run_graph_pipeline",
    "run_full_simulation",
    "run_full_pipeline"
]


def import_graph(graph_path, config, graph_type=None, input_edges=None):
    """Deserializes graph from disk. If no graph type is provided, it will\
    try to infer the type.

    Args:
        graph_path (str): File path to graph.
        config (light_config.LightConfig): Config for importing graph.
        graph_type (graph_types_pb2.GraphType, optional): Specifies format of the\
        serialized graph. Defaults to None.
        input_edges (list of lgf_pb2.EdgeInfo): Specifies the input edges\
        that will be used to run the graph. This helps with determining shapes\
        for graphs that might support unknown dimensions. Defaults to None.

    Returns:
        lgf_graph.LightGraph: Imported graph.
    """
    if graph_type is None:
        graph_type = graph_import.infer_import_graph_type(graph_path,
                                                          config,
                                                          input_edges=input_edges)

    graph_importer_cls = graph_import.get_graph_importer(graph_type)
    importer = graph_importer_cls(graph_path, config.sw_config, input_edges=input_edges)

    return importer.as_light_graph()


def transform_graph(graph, config, calibration_data=None):
    """Performs transformation and calibration on graph to optimize for OPU\
    hardware and/or simulation.

    Args:
        graph (lgf_graph.LightGraph): Graph to transform.
        config (light_config.LightConfig): Config for the transformations.
        calibration_data (inference_pb2.BatchedInferenceInput, optional): Data used for\
        calibration routines. If none provided, it is generated. Defaults to None.

    Returns:
        lgf_graph.LightGraph: Transformed graph.
    """
    return transforms.main(graph,
                           calibration_data,
                           config.hw_specs,
                           config.sw_config,
                           config.sim_params)


def export_graph(graph, path, config, graph_type=None):
    """Serializes graph to disk as the provided graph type. If none provided,\
    it will try to infer the graph type.

    Args:
        graph (lgf_graph.LightGraph): Graph to export.
        path (str): File path to save serialized graph.
        config (light_config.LightConfig): Config for exporter.
        graph_type (graph_types_pb2.GraphType, optional): specifies the format for\
        exporting the graph. Defaults to None.
    """
    if not graph_type:
        graph_type, exporter = graph_export.infer_export_graph_type(graph, config)
    else:
        exporter_cls = graph_export.get_graph_exporter(graph_type)
        exporter = exporter_cls(graph,
                                config.hw_specs,
                                config.sw_config,
                                config.sim_params)

    exporter.export_graph(path)


def run_functional_simulation(graph, inputs, config):
    """Runs a functional inference simulation on a graph with the provided input tensors.\
    Returns computed outputs. Requires LightConfig to set simulation parameters,\
    hardware specification, and software configuration in the simulation.

    Args:
        graph (lgf_graph.LightGraph): Graph to simulate.
        inputs (inference_pb2.BatchedInferenceInput): Input tensors.
        config (light_config.LightConfig): Configuration object. Must have\
        arch_type set to VIRTUAL.

    Returns:
        inference_pb2.BatchedInferenceOutput: Output tensors.
    """

    outputs = funcsim.simulate(graph, inputs, config)

    return outputs


def run_performance_simulation(graph, config):
    """Runs a performance simulation which simulates the number of total\
    clock cycles to run the graph.

    Args:
        graph (lgf_graph.LightGraph): Graph to simulate.
        config (light_config.LightConfig): Config for simulation.

    Returns:
        performance_data_pb2.ExecutionStats: Execution statistics for simulation.
    """
    perf_data = perfsim.simulate(graph, config.to_proto())

    return perf_data.execution_stats


def run_graph_pipeline(graph_path, config, export=False, export_path=""):
    """Imports, transforms and optionally exports graph in LightGraph format.

    Args:
        graph_path (str): File path to serialized graph.
        config (light_config.LightConfig): Config for graph pipeline.
        export (bool, optional): Enables export. Defaults to False.
        export_path (str, optional): Path to export transformed graph. Defaults to "".

    Returns:
        lgf_graph.LightGraph: Imported and transformed graph.
    """
    graph = import_graph(graph_path, config)
    transformed_graph = transform_graph(graph, config)

    if export:
        export_graph(transformed_graph, export_path, config.sw_config)

    return transformed_graph


def run_full_simulation(graph, data, config):
    """Runs both the functional and performance simulations.

    Args:
        graph (lgf_graph.LightGraph): Graph to simulate.
        data (inference_pb2.BatchInferenceInput): Input tensors for inference.
        config (light_config.LightConfig): Config for simulation.

    Returns:
        (performance_data_pb2.ExecutionStats, inference_pb2.BatchInferenceOutput):\
        Tuple containing the execution statistics for the performance simulation, and\
        the outputs of the functional inference simulation.

    """
    func_sim_results = run_functional_simulation(graph, data, config)
    perf_sim_results = run_performance_simulation(graph, config)

    return perf_sim_results, func_sim_results


def run_full_pipeline(graph_path, data, config, export=False, export_path=""):
    """Runs full pipeline which includes importing the graph, applying transforms,\
    optionally exporting, and running both functional and performance simulations.

    Args:
        graph_path (str): File path to serialized graph
        data (inference_pb2.BatchInferenceInput): Input tensors for inference.
        config (light_config.LightConfig): Config for pipeline.
        export (bool, optional): Enables graph export. Defaults to False.
        export_path (str, optional): Path to export graph. Defaults to "".

    Returns:
        (performance_data_pb2.ExecutionStats, inference_pb2.BatchInferenceOutput):\
        Tuple containing the execution statistics for the performance simulation, and\
        the outputs of the functional inference simulation.
    """
    graph = run_graph_pipeline(graph_path,
                               config,
                               export=export,
                               export_path=export_path)

    results = run_full_simulation(graph, data, config)

    return results
