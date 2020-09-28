import collections
import glob
import os
import shutil

import networkx as nx
import numpy as np

from sdk2.analysis import plot_layer_error
from sdk2.common import py_file_utils
from sdk2.graph import lgf_graph
from sdk2.graph.transform_graph import utils
from sdk2.graph.transform_graph.graph_transformers import (convert_to_debug_mode,
                                                           expand_subgraph_nodes)
from sdk2.proto import (calibration_pb2, dtypes_pb2, graph_types_pb2, hw_spec_pb2,
                        lgf_pb2, ops_pb2, performance_data_pb2, sim_params_pb2,
                        sw_config_pb2)

PROTO_ENUM_TO_FN_MAP = {
    "Op": ops_pb2.Op.Name,
    "GraphType": graph_types_pb2.GraphType.Name,
    "Padding": lgf_pb2.ImagePatchAttributes.Padding.Name,
    "DataFormat": lgf_pb2.ImagePatchAttributes.DataFormat.Name,
    "Type": dtypes_pb2.Type.Name,
}


def main(light_graph_pb_path,
         reduce_unsupported=False,
         reduce_while_nodes=True,
         list_of_pb_histogram_folder_paths=None,
         workload=None,
         error_output_dir=None):
    """
    This code converts protobuf files to json file ,
    adds dummy nodes and calculates appropriate positions for nodes for visualization.

    :param path_to_pb: path to proto file
    :return: graph dictionary which can be converted to json format.
    """

    graph_transform = expand_subgraph_nodes.ExpandSubgraphNodes(
        hw_spec_pb2.HardwareSpecs(), sw_config_pb2.SoftwareConfig(),
        sim_params_pb2.SimulationParams())

    # Transform light graph protobuf to light graph
    light_graph = graph_transform.process_transforms(
        lgf_graph.LightGraph.lgf_pb_to_graph(
            lgf_graph.LightGraph.read_lgf_pb(light_graph_pb_path)))

    # Nodes and link is a structure required by d3.js
    list_of_links = []
    list_of_nodes = []
    # Used to store i/p nodes that are input to the graph and are non-dummy nodes.
    input_nodes = set()
    node_set = set()

    # Simultaneously creating nx graph to get absolute node positions to space
    # them appropriately.
    nxGraph = nx.Graph()
    dummy_nodes_counter = 0

    for node in light_graph.nodes():
        for ctr, input_edge in enumerate(node.inputs):
            if light_graph.has_node(input_edge.name):
                """ To check if incoming node is input node by checking if it is in present
                in graph. """
                source = input_edge.name
            else:
                source = "Dummy" + str(dummy_nodes_counter)
                dummy_nodes_counter += 1
            target = node.name

            linkname = input_edge.name + str(input_edge.port)
            # Add other attributes for link here and append as argument
            if not reduce_unsupported or (
                    light_graph.get_node_by_name(source).supported
                    or light_graph.get_node_by_name(target).supported):
                # Add link only if reduce flag is not set or if
                # either src/tar is added (supported)
                if (not reduce_while_nodes
                        or "/while/" not in source and "/while/" not in target):
                    # Checking if while keyword in node names
                    node_set.add(target)
                    node_set.add(source)
                    list_of_links.append((source, target, linkname))

        for input_name in node.control_inputs:
            if not light_graph.has_node(input_name):
                raise ValueError("Could not find control input {0}".format(input_name))
            if not reduce_unsupported or light_graph.get_node_by_name(
                    input_name).supported:
                if not reduce_while_nodes or "/while/" not in node.name:
                    node_set.add(input_name)
                    list_of_links.append((input_name, node.name, "ctrl_input"))

    for edge in light_graph.output_edges():
        source = edge.name
        target = "Dummy" + str(dummy_nodes_counter)
        dummy_nodes_counter += 1
        linkname = edge.name + str(edge.port)
        # Add other attributes for link here and append as argument
        if not reduce_unsupported or light_graph.get_node_by_name(source).supported:
            list_of_links.append((source, target, linkname))
            node_set.add(source)
            node_set.add(target)

    for node in light_graph.input_edges():
        input_nodes.add(node.name)

    # list provides a better layout than set
    list_of_nodes = list(node_set)

    # Adding nodes and edges to nxGraph
    for node in list_of_nodes:
        nxGraph.add_node(node)

    for link in list_of_links:
        nxGraph.add_edge(link[0], link[1])

    positions_dict = nx.spectral_layout(nxGraph)
    # get position from spectral layout which is faster than others.

    graph = {}
    graph["nodes"] = []
    graph["links"] = []
    graph["directories"] = ["Graph"]
    graph["workload"] = []
    graph["op_type"] = []
    graph["all_opu_type_names"] = []

    node_name_to_error_dict = collections.defaultdict(int)
    if workload:
        make_tmp_dir = not error_output_dir
        if make_tmp_dir:
            error_output_dir = py_file_utils.mkdtemp()
        fname = os.path.join(error_output_dir, "relative_error_data.pb")

        if not os.path.exists(fname):
            # Writes the pb
            plot_layer_error.main(workload, error_output_dir, fix_bns=False)

        # Reads the pb
        errors_pb = performance_data_pb2.RelativeErrorData()
        with open(fname, "rb") as f:
            errors_pb.ParseFromString(f.read())
        node_name_to_error_dict = collections.defaultdict(int,
                                                          dict(errors_pb.errors_dict))
        for name, error in node_name_to_error_dict.items():
            graph["workload"].append({"name": name, "error": error})

        if make_tmp_dir:
            shutil.rmtree(error_output_dir)

    opu_type_set = set()
    for index, node in enumerate(node_set):
        group = -1
        hover_text = ""
        error_value = 0
        opu_type_name = light_graph.get_node_by_name(node).WhichOneof(
            "node") if not node.startswith("Dummy") else "Dummy"
        opu_type_set.add(opu_type_name)

        if not node.startswith("Dummy"):
            group = graph_transform.node_name_to_subgraph_id(node) \
                if (light_graph.get_node_by_name(node).supported) else -1
            if group is None:
                group = -1
            lnf_node = light_graph.get_node_by_name(node)
            hover_text = text_list_to_hover_text(
                message_to_text_list(getattr(lnf_node, lnf_node.WhichOneof("node"))))
            error_value = node_name_to_error_dict[node]

        graph["nodes"].append({
            "name": node,
            "node_info": opu_type_name,
            "group": group + 1,
            "hover_text": hover_text,
            "error": error_value
        })
        # Take position from the layout algorithm.
        graph["nodes"][index]["x"] = positions_dict[node][0]
        graph["nodes"][index]["y"] = positions_dict[node][1]

    graph["all_opu_type_names"] = list(opu_type_set)
    name_to_hist_dict = collections.defaultdict(list)

    if list_of_pb_histogram_folder_paths:
        for folder_path in list_of_pb_histogram_folder_paths:
            graph["directories"].append(folder_path)

        # If path exists add the histogram.pb files as hist arrays to dict
        fetch_all_histograms(list_of_pb_histogram_folder_paths, name_to_hist_dict)

    # Excluding dummy nodes since no info is associated with dummy nodes
    for link in list_of_links:
        histograms_per_edge = []
        # Unsupported edges have port as -1
        port = int(link[2][-1]) if not link[2].startswith("ctrl_input") else -1

        if not link[0].startswith("Dummy"):
            # Add histogram data if the source node is a const node.
            if get_tensor_data_from_const_node(light_graph.get_node_by_name(link[0])):
                histograms_per_edge.append(
                    get_tensor_data_from_const_node(light_graph.get_node_by_name(
                        link[0])))

            # check in dictionary if current blank histograms are supposed to be filled
            if not histograms_per_edge:
                node_file_name = (
                    convert_to_debug_mode.ConvertToDebugMode.file_friendly_name(
                        link[2][:-1]))
                names_to_try = [node_file_name]
                if "_0_cast" in node_file_name:
                    names_to_try.append(node_file_name.strip("_0_cast"))
                for n in names_to_try:
                    histograms_per_edge = name_to_hist_dict[n + ":" + link[2][-1]]
                    if histograms_per_edge:
                        break

        graph["links"].append({
            "source": link[0],
            "target": link[1],
            "linkname": link[0] + ":" + str(port),
            "hover_info": get_hover_text_link(link, light_graph, port),
            "edge_hist_data": histograms_per_edge or [],
            "port": port
        })

    graph["input_nodes"] = []
    for node in input_nodes:
        graph["input_nodes"].append({"name": node})

    # Adding all histogram data from folder in the graph.

    # To print the graph (dictionary) use this: print(json.dumps(graph))
    return graph


def singular_value_to_string(descr, val, extract_binary=False):
    enum_name = None if descr.enum_type is None else descr.enum_type.name
    return PROTO_ENUM_TO_FN_MAP.get(enum_name, lambda x: str(x))(val)


def edge_info_to_text_list(edge_info):
    return [("Name", edge_info.name), ("Port", edge_info.port),
            ("DType", "{0} {1}".format(PROTO_ENUM_TO_FN_MAP["Type"](edge_info.dtype.t),
                                       edge_info.dtype.p)),
            ("Shape", "{0}, {1}".format(edge_info.shape.d,
                                        edge_info.shape.batch_dim_indx))]


def get_hover_text_link(link, light_graph, port):
    if port == -1:
        return ""
    if link[0].startswith("Dummy"):
        return "Dummy"
    else:
        return text_list_to_hover_text(
            edge_info_to_text_list(light_graph.get_node_by_name(link[0]).outputs[port]))


def get_tensor_data_from_const_node(node):
    # Fetches content to display on node hover
    if node.HasField(lgf_pb2.LNF.const.DESCRIPTOR.name):
        tensor_pb = node.const.value
    elif node.HasField(lgf_pb2.LNF.variable.DESCRIPTOR.name):
        tensor_pb = node.variable.const.value
    else:
        return None

    # Converting numpy_array to python array to make it JSON serializable

    array_with_zeroes = utils.tensor_pb_to_array(tensor_pb, np.float64).flatten()
    if np.count_nonzero(array_with_zeroes) == 0:
        return [[len(array_with_zeroes)], 0]

    array_without_zeroes = array_with_zeroes[np.nonzero(array_with_zeroes)]

    max_abs_val = np.max(np.abs(array_without_zeroes))
    counts, _ = np.histogram(array_without_zeroes,
                             bins=20 if len(array_without_zeroes) < 4096 else 4096,
                             range=(-max_abs_val, max_abs_val))

    # All histograms already present inside the graph have directory number as -1
    return [
        counts.tolist(),
        np.float64(max_abs_val),
        len(array_with_zeroes) - len(array_without_zeroes), -1
    ]


def message_to_text_list(msg):
    # Convert msg to a text list
    text_list = []
    ignore_set = set(["hist_keys_before_adc", "hist_keys_after_adc", "value"])
    for _, descr in msg.DESCRIPTOR.fields_by_name.items():
        val = getattr(msg, descr.name)
        if descr.name not in ignore_set:
            if hasattr(val, "ListFields"):
                # val is a message, so recursively convert to text list
                text_list.append((descr.name, message_to_text_list(val)))

            else:
                # val is a singular value like int, float, string, list, etc.
                text_list.append((descr.name, singular_value_to_string(descr, val)))

    return text_list


def text_list_to_hover_text(text_list, tabs=0):
    """
    Params:
        text_list: a list of (k, v) tuples, where k is a string a v is either
            a string or another text_list
        tabs: number of tabs to indent the text

    Returns:
        a string that can be used for hover text, if v is another text_list it
            is indented by recursively caling this function with an extra tab
    """
    hover_text = []
    for k, v in text_list:
        if isinstance(v, list):
            v = text_list_to_hover_text(v, tabs=tabs + 1)
            if len(v) > 0:
                v = "<br>{0}".format(v)

        space = " " * 4 * tabs
        hover_text.append("{0}<b>{1}</b>: {2}".format(space, k, v))

    return "<br>".join(hover_text)


def fetch_all_histograms(list_of_pb_histogram_folder_paths, name_to_hist_dict):
    for directory_number, folder_path in enumerate(list_of_pb_histogram_folder_paths):
        # directory number is later used to color the hist based on dir.
        if not os.path.exists(folder_path):
            raise Exception("Not a valid path")
        # Appending forward-slash based on whether or not user added it in path
        # and getting filename from splitting the string between "/" and ".pb"
        pb_histograms_list = glob.glob(os.path.join(folder_path, "*.pb"), recursive=True)
        for filename in pb_histograms_list:
            cal_hist = calibration_pb2.CalibrationHistogram()
            with open(filename, "rb") as f:
                cal_hist.ParseFromString(f.read())
            counts = list(cal_hist.hist)
            counts = [min(c, 1e6) for c in counts]
            max_size = cal_hist.value[calibration_pb2.CalibrationHistogram.HISTOGRAM_MAX]
            zeroes = cal_hist.value[calibration_pb2.CalibrationHistogram.NUM_ZEROS]
            name_to_hist_dict[filename.split("/")[-1].split(".pb")[0]].append(
                [counts, max_size, zeroes, directory_number])
