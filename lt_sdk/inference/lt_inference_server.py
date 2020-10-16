import argparse
import logging
from concurrent import futures

import grpc

from lt_sdk.graph import lgf_graph
from lt_sdk.graph.run_graph import graph_runner
from lt_sdk.proto import (
    graph_types_pb2,
    hardware_configs_pb2,
    inference_pb2,
    inference_pb2_grpc,
)
from lt_sdk.proto.configs import config


class LTInferenceServer(inference_pb2_grpc.LTInferenceServicer):
    """gRPC server wrapping graph_runner."""

    def __init__(self, lgf_path, hw_cfg=hardware_configs_pb2.DELTA):
        self._lg = lgf_graph.LightGraph.from_pb(lgf_path)
        spec, sw, sim = config.get_config(hw_cfg, graph_types_pb2.LGFProtobuf)
        self._runner = graph_runner.GraphRunner(self._lg, spec, sw, sim)

    def GetInputSpec(self, request, context):
        ret = inference_pb2.GetInputSpecResponse()
        for e in self._lg.input_edges():
            ret.inputs.add().CopyFrom(e)
        return ret

    def Predict(self, request, context):
        return self._runner.run(request)

    def PredictBatch(self, request, context):
        return self._runner.run_single_batch(request)


def serve(lgf_path, port, hw_cfg):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_LTInferenceServicer_to_server(
        LTInferenceServer(lgf_path,
                          hw_cfg=hw_cfg),
        server)
    server.add_insecure_port("[::]:{0}".format(port))
    server.start()
    logging.info("LT Inference Server started ...")
    server.wait_for_termination()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, help="port for gRPC service", default=6000)
    parser.add_argument("--lgf_path", type=str, help="path to a lgf pb", default=None)
    parser.add_argument("--config_name",
                        type=str,
                        help="path to a lgf pb",
                        default=hardware_configs_pb2.HardwareConfig.Name(
                            hardware_configs_pb2.DELTA))

    logging.basicConfig()
    args = parser.parse_args()
    serve(args.lgf_path,
          args.port,
          hardware_configs_pb2.HardwareConfig.Value(args.config_name))


if __name__ == "__main__":
    main()
