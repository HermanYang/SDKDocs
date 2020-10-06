import argparse
import glob
import logging
import os
import shutil

import numpy as np
from plotly import graph_objs as go
from plotly import offline as py

from lt_sdk.common import py_test_util
from lt_sdk.proto import calibration_pb2


def get_hist_and_bin_edges(cal_hist_pb):
    # Assumes hist_pb has populated histograms
    assert (len(cal_hist_pb.hist) > 0)
    hist = np.array(cal_hist_pb.hist).astype(np.float64)

    # Create bin_edges
    max_val = cal_hist_pb.value[calibration_pb2.CalibrationHistogram.HISTOGRAM_MAX]
    bin_size = (2 * max_val) / len(cal_hist_pb.hist)
    bin_edges = np.arange(-max_val, max_val + bin_size, bin_size, np.float64)

    # Clip ends of hist and bin_edges if possible
    non_zero_locations = np.where(hist != 0)[0]
    extra = -1
    if len(non_zero_locations) > 0:
        start_edge = non_zero_locations[0]
        end_edge = non_zero_locations[-1]
        num_bins = hist.shape[0]
        extra = min(start_edge, num_bins - end_edge)

    if extra > 0 and extra < hist.shape[0] // 2:
        # Clip the histogram if it won"t make it empty
        hist = hist[extra:-extra]
        bin_edges = bin_edges[extra:-extra]

    return hist, bin_edges


def cal_hist_pb_to_plotly_fig(cal_hist_pb, plot_title=""):
    # Get counts and bin edges
    hist, bin_edges = get_hist_and_bin_edges(cal_hist_pb)
    num_zeros = cal_hist_pb.value[calibration_pb2.CalibrationHistogram.NUM_ZEROS]

    # Create simple bar chart
    bin_width = bin_edges[1] - bin_edges[0]
    bar_chart = go.Bar(x=bin_edges[:-1], y=hist, width=bin_width)

    # Decrease height of the plot if there is one bin that is much
    # larger than all the others
    sorted_counts = sorted(hist)
    y_range = None
    if sorted_counts[-1] > 1.25 * sorted_counts[-2]:
        y_range = (0, 1.25 * sorted_counts[-2])

    layout = go.Layout(
        title=dict(text=plot_title),
        xaxis=dict(title=dict(text="bins (excluding {} zeros)".format(num_zeros))),
        yaxis=dict(title=dict(text="count"),
                   range=y_range),
    )

    data = [bar_chart]
    fig = go.Figure(data=data, layout=layout)

    return fig


def main(cal_hist_pb_map, output_dir, plot_title_map={}):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    logging.info("Histogram plots saved at:")
    for key, cal_hist_pb in cal_hist_pb_map.items():
        plot_title = plot_title_map.get(key, str(key))
        fig = cal_hist_pb_to_plotly_fig(cal_hist_pb, plot_title=plot_title)

        plot_path = os.path.join(output_dir, "{}.html".format(plot_title))
        py.plot(fig, auto_open=False, filename=plot_path)
        logging.info("file://{}".format(plot_path))


def _add_to_map(cal_hist_pb_map, cal_hist_pb_path):
    key = os.path.basename(cal_hist_pb_path)[:-3]
    cal_hist_pb = calibration_pb2.CalibrationHistogram()
    with open(cal_hist_pb_path, "rb") as f:
        cal_hist_pb.ParseFromString(f.read())
    cal_hist_pb_map[key] = cal_hist_pb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hist_dir",
                        type=str,
                        help="path a directory of histogram protobufs")
    parser.add_argument("--hist_path",
                        default=None,
                        type=str,
                        help="path to a single histogram protobuf")
    parser.add_argument("--output_dir",
                        type=str,
                        help="directory to store html output files")
    args = parser.parse_args()

    py_test_util.PythonTestProgram.set_root_logger(logging_level=logging.INFO,
                                                   logging_format="%(message)s")

    # Create a map from the hist_dir or hist_path
    cal_hist_pb_map = {}
    if args.hist_path:
        _add_to_map(cal_hist_pb_map, args.hist_path)
    else:
        for cal_hist_pb_path in glob.glob(os.path.join(args.hist_dir, "*.pb")):
            _add_to_map(cal_hist_pb_map, cal_hist_pb_path)

    main(cal_hist_pb_map, args.output_dir)
