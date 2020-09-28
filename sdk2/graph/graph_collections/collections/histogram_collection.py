import ctypes
import os

from sdk2.proto import calibration_pb2, common_pb2
from sdk2.visuals import plot_histograms


class HistogramCollection(object):
    """
    Contains a map from integers to calibration_pb2.CalibrationHistogram() objects

    Wrapper around a cpp class
    Instances of this class should only be created by a
    graph_collection.GraphCollection() object
    """

    lib = ctypes.CDLL(
        os.path.join(os.path.dirname(__file__), "lib_histogram_collection.so"))

    lib.GetHistogram.argtypes = (ctypes.c_void_p, ctypes.c_uint,
                                 ctypes.POINTER(ctypes.c_uint))
    lib.GetHistogram.restype = ctypes.POINTER(ctypes.c_char)

    lib.InitializeEmptyHistogram.argtypes = (ctypes.c_void_p, ctypes.c_uint,
                                             ctypes.c_uint)

    lib.GetHistogramMode.argtypes = (ctypes.c_void_p, ctypes.c_uint)
    lib.GetHistogramMode.restype = ctypes.c_uint

    lib.UpdateHistogramMode.argtypes = (ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint)

    lib.ComputeQuantScales.argtypes = (ctypes.c_void_p, ctypes.c_uint,
                                       ctypes.POINTER(ctypes.c_char), ctypes.c_uint,
                                       ctypes.POINTER(ctypes.c_char),
                                       ctypes.c_uint, ctypes.c_uint,
                                       ctypes.POINTER(ctypes.c_uint), ctypes.c_bool)
    lib.ComputeQuantScales.restype = ctypes.POINTER(ctypes.c_char)

    lib.HistogramCollectionGetKeys.argtypes = (ctypes.c_void_p,
                                               ctypes.POINTER(ctypes.c_uint))
    lib.HistogramCollectionGetKeys.restype = ctypes.POINTER(ctypes.c_char)

    lib.ClearProtoString.argtypes = (ctypes.c_void_p,)

    lib.GetHistogramMaxVal.argtypes = (ctypes.c_void_p, ctypes.c_uint)
    lib.GetHistogramMaxVal.restype = ctypes.c_double

    lib.SetHistogramMaxVal.argtypes = (ctypes.c_void_p, ctypes.c_uint, ctypes.c_double)

    def __init__(self, pointer):
        self._obj = pointer

    def get_histogram(self, key):
        histogram_size = ctypes.c_uint()
        histogram_ptr = self.lib.GetHistogram(self._obj, key,
                                              ctypes.byref(histogram_size))

        cal_hist_pb = calibration_pb2.CalibrationHistogram()
        cal_hist_pb.ParseFromString(histogram_ptr[:histogram_size.value])

        self.lib.ClearProtoString(self._obj)

        return cal_hist_pb

    def initialize_empty_histogram(self, key, num_bins):
        self.lib.InitializeEmptyHistogram(self._obj, key, num_bins)

    def get_histogram_mode(self, key):
        return self.lib.GetHistogramMode(self._obj, key)

    def update_histogram_mode(self, key, mode):
        self.lib.UpdateHistogramMode(self._obj, key, mode)

    def get_quant_scales(self,
                         quant_method,
                         sw_config,
                         precisions,
                         bias_type,
                         use_unsigned_quant_scheme=False):
        precisions_data = precisions.SerializeToString()
        sw_config_data = sw_config.SerializeToString()
        quant_scales_size = ctypes.c_uint()

        quant_scales_ptr = self.lib.ComputeQuantScales(self._obj,
                                                       quant_method, sw_config_data,
                                                       len(sw_config_data),
                                                       precisions_data,
                                                       len(precisions_data), bias_type,
                                                       ctypes.byref(quant_scales_size),
                                                       use_unsigned_quant_scheme)

        quant_scales = calibration_pb2.ScaleInfoMap()
        quant_scales.ParseFromString(quant_scales_ptr[:quant_scales_size.value])

        self.lib.ClearProtoString(self._obj)

        return quant_scales

    def get_keys(self):
        keys_size = ctypes.c_uint()
        keys_ptr = self.lib.HistogramCollectionGetKeys(self._obj,
                                                       ctypes.byref(keys_size))

        keys = common_pb2.Param()
        keys.ParseFromString(keys_ptr[:keys_size.value])

        self.lib.ClearProtoString(self._obj)

        return list(keys.l.i)

    def get_histogram_max_val(self, key):
        return self.lib.GetHistogramMaxVal(self._obj, key)

    def set_histogram_max_val(self, key, max_val):
        self.lib.SetHistogramMaxVal(self._obj, key, max_val)

    def plot_histograms(self, output_dir, plot_title_map={}):
        keys = self.get_keys()
        cal_hist_pb_map = {k: self.get_histogram(k) for k in keys}
        plot_histograms.main(cal_hist_pb_map, output_dir, plot_title_map=plot_title_map)
