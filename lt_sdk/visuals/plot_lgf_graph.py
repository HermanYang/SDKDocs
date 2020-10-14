import argparse

from graph_visualization import create_flask_app
from lt_sdk.visuals import lgf_proto_to_json


def main():
    """ Runs a web server based on flask, specifying port is optional
    (defaulted to 5000) """
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, help="path to a lgf pb", default=5000)
    parser.add_argument("--pb_graph_path", type=str, help="path to a pb graph file")
    parser.add_argument("--pb_histogram_folder_path",
                        type=str,
                        help="path to a histogram folder containing protobuf files",
                        action="append")
    parser.add_argument("--reduce_unsupported",
                        action="store_true",
                        help=("reduces unsupported nodes by deleting them if"
                              "it is a unsupp -> unsupp connection"),
                        default=False),
    parser.add_argument("--reduce_while_nodes",
                        action="store_true",
                        help=("removes all '/while/' nodes by deleting them"
                              "before passing it over to json file"),
                        default=True)
    parser.add_argument(
        "--workload",
        type=str,
        help="workload name, must be a string in performance_sweep_map.py")
    parser.add_argument("--error_output_dir",
                        type=str,
                        help="Output directory for storing error pb")
    parser.add_argument("--no_text", default=False, action="store_true")
    args = parser.parse_args()
    json_object = lgf_proto_to_json.main(
        args.pb_graph_path,
        reduce_unsupported=args.reduce_unsupported,
        reduce_while_nodes=args.reduce_while_nodes,
        list_of_pb_histogram_folder_paths=args.pb_histogram_folder_path,
        workload=args.workload,
        error_output_dir=args.error_output_dir)

    create_flask_app.run_flask(args.port, json_object)


if __name__ == "__main__":
    main()
