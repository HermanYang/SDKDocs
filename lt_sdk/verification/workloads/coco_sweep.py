import os

import numpy as np

from lt_sdk.graph.transform_graph import utils
from lt_sdk.proto import graph_types_pb2, node_filters
from lt_sdk.verification.tools import object_detection_utils
from lt_sdk.verification.workloads import model_quality, standard_data_base


class COCOSweep(standard_data_base.StandardDataBase,
                model_quality.ObjectDetectionWorkload):

    TEST_LABELS_DIR_NAME = "test_labels"
    TEST_IMAGES_SIZES_FILE_NAME = "test_images_sizes.npy"

    def base_data_dir(self):
        return "coco_data"

    def graph_type(self):
        return graph_types_pb2.TFSavedModel

    def compilation_batch_size(self):
        return 2

    def py_batch_size(self):
        return 160

    def input_names(self):
        return ["inputs"]

    def original_image_size(self):
        return [416, 416]

    def prediction_edge(self):
        return "output_boxes"

    def get_test_labels(self, shard):
        assert (shard == 0)
        test_labels_dir = os.path.join(self._data_dir, self.TEST_LABELS_DIR_NAME)
        test_images_sizes = np.load(
            os.path.join(self._data_dir,
                         self.TEST_IMAGES_SIZES_FILE_NAME))
        return test_labels_dir, test_images_sizes


class YOLOV3Sweep(COCOSweep):

    def graph_dir(self):
        return "yolov3"


class YOLOV3TinySweep(COCOSweep):

    def graph_dir(self):
        return "yolov3_tiny"


class ObjDectModelZooSweep(COCOSweep):
    """Models that came from TF Model Zoo."""

    def input_names(self):
        return ["image_tensor"]

    def prediction_edge(self):
        return "detection_boxes"

    def inference_to_obj_det_results(self, inf_out, test_images_sizes, start_img):
        results = []
        arrays = {}

        BOXES = "detection_boxes"
        CLASSES = "detection_classes"
        SCORES = "detection_scores"
        NUM = "num_detections"

        for named_tensor in inf_out.results:
            arrays[named_tensor.edge_info.name] = utils.tensor_pb_to_array(
                named_tensor.data,
                utils.dtype_pb_to_np_dtype(named_tensor.edge_info.dtype))

        orig_size = self.original_image_size()
        for j in range(arrays[BOXES].shape[0]):
            label_ind = start_img + j
            detection_result = model_quality.ObjDetResult(label_ind)
            for i in range(int(arrays[NUM][j])):
                box = arrays[BOXES][j, i, :]
                # scale to size in pixels
                new_box = np.array([
                    box[0] * orig_size[0],
                    box[1] * orig_size[1],
                    box[2] * orig_size[0],
                    box[3] * orig_size[1]
                ])
                new_box = object_detection_utils.convert_to_original_size(
                    new_box,
                    np.array(self.original_image_size()),
                    test_images_sizes[label_ind],
                    True)
                detection_result.add_box(
                    model_quality.BoundingBox(arrays[CLASSES][j,
                                                              i],
                                              arrays[SCORES][j,
                                                             i],
                                              new_box))

            results.append(detection_result)

        return results


class FasterRCNNResNet50Sweep(ObjDectModelZooSweep):

    def graph_dir(self):
        return "faster_rcnn_resnet50"

    def ignore_nodes_filter(self):
        keep_nodes = node_filters.or_filter(
            node_filters.name_starts_with_filter(
                "FirstStageFeatureExtractor/resnet_v1_50"),
            node_filters.name_starts_with_filter(
                "SecondStageFeatureExtractor/resnet_v1_50"))
        return node_filters.not_filter(keep_nodes)


class SSDResNet50Sweep(ObjDectModelZooSweep):

    def graph_dir(self):
        return "ssd_resnet50"

    def ignore_nodes_filter(self):
        keep_nodes = [
            node_filters.name_starts_with_filter("FeatureExtractor/resnet_v1_50"),
            node_filters.name_starts_with_filter(
                "WeightSharedConvolutionalBoxPredictor"),
        ]
        keep_nodes = node_filters.or_filter(*keep_nodes)
        return node_filters.not_filter(keep_nodes)
