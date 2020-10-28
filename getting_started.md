# Getting Started

This guide will go over the basic functionality available in the SDK. This includes tasks such as:

- Importing a graph from disk
- Applying transforms and optimizations to the graph
- Exporting a graph to disk
- Running a performance simulation for measuring clock cycles
- Running a functional simulation for graph inference

## Using the SDK

To use the SDK, import the package as follows:

```python
import lt_sdk as lt
```

We use the alias `lt` when importing the package.

## Create Default Config Object

For most use cases, you will want to use the default configuration which is set for the Delta hardware configuration. The following will create a config that can be used for working with graphs on the Delta archictecture. See the API documentation for more details.

```python
config = lt.get_default_config()
```

> This `config` object will be used throughout the rest of the guide.

## Importing a Graph from Disk

Do the following to import an existing graph/model from disk:

```python
graph_path = "/path/to/model"

imported_graph = lt.import_graph(graph_path, config)
```

This will import the model if it is saved in one of the supported formats. We currently fully support TFSavedModel and Light Graph Format (LGF). We currently partially support ONNX, but are planning to add full support for ONNX, TFLite, and PyTorch in the future.

## Applying Transformations & Optimizations to the Graph

For most graphs, we will need to apply a standard set of transformations to optimize them to run on the OPU and simulator. These transformations can be applied in the following manner:

```python
light_graph = lt.transform_graph(imported_graph, config)
```

## Exporting a Graph to Light Graph Format (LGF)

Do the following to save your model to disk for later use in Light Graph Format (LGF):

```python
output_path = "/path/to/output/graph_lgf.pb"

lt.export_graph(light_graph, output_path, config)
```

This can be imported later using the method described in [Importing a Graph from Disk](#importing-a-graph-from-disk).

## Performance Simulation

Performance simulation estimates the total number of clock cycles necessary to run the graph on the provided hardware configuration.

```python
execution_stats = lt.run_performance_simulation(light_graph, config)

```

`execution_stats` contains the total clock cycles along with other metadata about the simulation. This protobuf can then be converted to a chrome tracing file with the following:

```python
from lt_sdk.visuals import sim_result_to_trace

filepath = "/path/to/output.trace"

sim_result_to_trace.instruction_trace(filepath, execution_stats,
                                      config.hw_specs, config.sim_params)
```

## Functional Simulation

A functional simulation will run the full computation graph and produce a functionally correct output. This type of simulation is used to validate that the hardware will produce the desired output and metrics.

### Create Input Batch

We first want to create some placeholder input data to run through a graph.

```python
import numpy as np

batch_size = 1

#  Corresponds to input name in graph
input_name = "input"

# Random data to pass through graph
tensor = np.random.random((10, 300, 300, 3))

# Creates an object that maps a node name to a tensor.
# This can contain multiple tensors in the case of multiple inputs.
named_tensors = lt.data.NamedTensorSet([input_name], [tensor])

input_batch = lt.data.batch_inputs(named_tensors, batch_size)
```

### Run Functional Simulation

Next, we will use the placeholder data to run the functional simulation.

```python
# uses input batch created in previous section
outputs = lt.run_functional_simulation(light_graph, input_batch, config)
```

`outputs` will be the output tensor of the graph.
