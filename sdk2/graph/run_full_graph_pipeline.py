import argparse
import logging

from sdk2.common import py_test_util
from sdk2.graph import full_graph_pipeline
from sdk2.proto import graph_types_pb2, hardware_configs_pb2
from sdk2.proto.configs import config


def main_helper(input_path, output_dir, hw_cfg, graph_type):
    hw_specs, sw_config, sim_params = config.get_config(hw_cfg, graph_type)
    full_graph_pipeline.main(input_path, graph_type, output_dir,
                             graph_types_pb2.LGFProtobuf, None, hw_specs, sw_config,
                             sim_params)


def main():
    """Args are put in this function so that it's easy to make it an entry-point
    during sdk-deployment.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="file or dir path")
    parser.add_argument("--output_path", type=str, help="path to lgf format.")
    parser.add_argument("--config_name",
                        type=str,
                        default=hardware_configs_pb2.HardwareConfig.Name(
                            hardware_configs_pb2.DELTA),
                        help="output dir")
    parser.add_argument("--graph_type",
                        type=str,
                        default="tf",
                        help="name of input graph format")

    args = parser.parse_args()

    py_test_util.PythonTestProgram.set_root_logger(logging_level=logging.INFO)

    # TODO: automatically figure out import graph type
    main_helper(args.input_path, args.output_path,
                hardware_configs_pb2.HardwareConfig.Value(args.config_name),
                config.get_graph_type(args.graph_type))


if __name__ == "__main__":
    main()
