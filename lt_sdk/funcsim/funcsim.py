from lt_sdk.graph.graph_collections import graph_collection
from lt_sdk.graph.run_graph import graph_runner
from lt_sdk.proto import sim_params_pb2


def simulate(graph, inputs, config):
    arch_type = config.sim_params.arch_params.arch_type
    if arch_type != sim_params_pb2.ArchitectureParams.VIRTUAL:
        raise ValueError("arch_type must be VIRTUAL for simulation.")
    with graph_collection.GraphCollection() as graph_coll:
        sw_config, hw_spec, sim_params = config.get_configs()
        runner = graph_runner.GraphRunner(graph,
                                          hw_spec,
                                          sw_config,
                                          sim_params,
                                          graph_coll=graph_coll)

        outputs = runner.run(inputs)

    return outputs
