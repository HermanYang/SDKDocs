import logging

from sdk2.graph.transform_graph import cached_pipeline, utils
from sdk2.graph.transform_graph.calibration import (activation_scale_calibration,
                                                    adc_scale_calibration,
                                                    add_activation_scales,
                                                    add_adc_scales)
from sdk2.graph.transform_graph.graph_transformers import (add_cast_nodes,
                                                           apply_node_map,
                                                           fold_phasify_constants,
                                                           mark_supported_const,
                                                           remove_pad_nodes)
from sdk2.proto import inference_pb2, sw_config_pb2


class BaseTransformsStage(cached_pipeline.PipelineStage):

    def __init__(self, stage_args):
        super().__init__("SoftwareConfig BaseTransforms", stage_args)

    def execute(self, light_graph):
        # Software Config Transforms
        node_map = apply_node_map.ApplyNodeMap.get_node_map_from_filter_transform_map(
            self._sw_config.filter_transform_map, self._hw_specs, self._sw_config,
            self._sim_params)
        graph_transformer = apply_node_map.ApplyNodeMap(self._hw_specs, self._sw_config,
                                                        node_map)
        light_graph = graph_transformer.process_transforms(light_graph, prune=True)

        # Convert constant nodes to supported if possible
        graph_transformer = mark_supported_const.MarkSupportedConstNodes(
            self._hw_specs, self._sw_config, self._sim_params)
        light_graph = graph_transformer.process_transforms(light_graph, prune=False)

        # Add cast nodes to resolve inconsistent dtypes
        graph_transformer = add_cast_nodes.AddCastNodes(self._sw_config)
        light_graph = graph_transformer.process_transforms(light_graph)

        # Remove unnecessary pad nodes
        graph_transformer = remove_pad_nodes.RemovePadNodes(self._hw_specs,
                                                            self._sw_config,
                                                            self._sim_params)
        return graph_transformer.process_transforms(light_graph)


class ActivationScaleCalibration(cached_pipeline.PipelineStage):

    def __init__(self, stage_args):
        super().__init__("Activation Scale Calibration", stage_args)
        self._cal_data = stage_args.calibration_data
        self._original_graph = stage_args.original_graph

    def execute(self, light_graph):
        # Get nodes to calibrate
        nodes_to_calibrate = add_activation_scales.AddActivationScales.\
            get_nodes_to_calibrate(light_graph, self._sw_config)

        utils.log_message("Activation Scale Calibration")
        activation_scales_data = activation_scale_calibration.main(
            self._original_graph, self._cal_data, self._hw_specs, self._sw_config,
            self._sim_params, nodes_to_calibrate)

        utils.log_message("Add Activation Scales")
        graph_transformer = add_activation_scales.AddActivationScales(
            self._hw_specs, self._sw_config, self._sim_params, activation_scales_data)
        return graph_transformer.process_transforms(light_graph, prune=True)


class ADCScaleCalibration(cached_pipeline.PipelineStage):
    """We break this out so it's easy to access as a cached stage."""

    def __init__(self, stage_args):
        super().__init__("ADC Scale Calibration", stage_args)
        self._cal_data = stage_args.calibration_data

    def _add_stage(self, stages, stage_name, output_graph):
        stage_args = cached_pipeline.StageArgs(self._cal_data, None, self._hw_specs,
                                               self._sw_config, self._sim_params)
        if output_graph is not None:
            stages.append(
                cached_pipeline.FnStage(
                    lambda x: output_graph,  # just return the output graph
                    "{0} {1}".format(self.name, stage_name),
                    stage_args))

    def execute(self, light_graph):
        utils.log_message("ADC Scale Calibration")
        adc_scales_data, calib_graph = adc_scale_calibration.main(
            light_graph, self._cal_data, self._hw_specs, self._sw_config,
            self._sim_params)

        utils.log_message("Add ADC Scales")
        graph_transformer = add_adc_scales.AddADCScales(self._hw_specs, self._sw_config,
                                                        self._sim_params,
                                                        adc_scales_data)
        light_graph = graph_transformer.process_transforms(light_graph)

        # Using a cached pipeline here just to save intermediate stages,
        # stages do not actually do any computation or save time with caching
        stages = []
        self._add_stage(stages, "Calib Graph", calib_graph)
        self._add_stage(stages, "Final Graph", light_graph)
        pipeline = cached_pipeline.CachedPipeline(stages)
        return pipeline.transform(light_graph)


class FoldPhasifyConstants(cached_pipeline.PipelineStage):
    """Folds constants related to phasify if possible"""

    def __init__(self, stage_args):
        super().__init__("Fold Phasify Constants", stage_args)

    def execute(self, light_graph):
        graph_transformer = fold_phasify_constants.FoldPhasifyConstants(
            self._hw_specs, self._sw_config, self._sim_params)
        return graph_transformer.process_transforms(light_graph)


STAGE_MAP = {
    sw_config_pb2.BASE_TRANSFORMS: BaseTransformsStage,
    sw_config_pb2.ACTIVATION_SCALE_CALIBRATION: ActivationScaleCalibration,
    sw_config_pb2.ADC_SCALE_CALIBRATION: ADCScaleCalibration,
    sw_config_pb2.FOLD_PHASIFY_CONSTANTS: FoldPhasifyConstants,
}


def main(light_graph, calibration_data, hw_specs, sw_config, sim_params):
    """
    Transforms a light graph from a graph_importer into a light graph
    that can run on our inference server
    Performs a set of standard transformations and calibration routines

    Params:
        light_graph: original LightGraph object, output from
            GraphImporter.as_light_graph()
        calibration_data: a inference_pb2.BatchedInferenceInput() protobuf (if None
            generate random data)
        hw_specs: a hw_spec_pb2.HardwareSpecs() protobuf
        sw_config: a sw_config_pb2.SoftwareConfig() protobuf
        sim_params: a sim_params_pb2.SimulationParams() protobuf

    Returns:
        A LightGraph that can be run on our inference server
    """
    if calibration_data is None:
        calibration_data = inference_pb2.BatchedInferenceInput()
        calibration_data.batches.add().CopyFrom(
            utils.generate_random_inference_inputs(
                light_graph.input_edges(),
                unknown_dim_size=sim_params.compiled_batch_size))

    if sw_config.cache_dir:
        logging.info("Using cache dir: {0}".format(sw_config.cache_dir))

    stage_args = cached_pipeline.StageArgs(calibration_data, light_graph, hw_specs,
                                           sw_config, sim_params)
    stages = [
        STAGE_MAP[stage](stage_args) for stage in sw_config.standard_transform_stages
    ]
    pipeline = cached_pipeline.CachedPipeline(stages)
    return pipeline.transform(light_graph)
