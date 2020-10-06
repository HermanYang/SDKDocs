import logging
import os
import shutil

import numpy as np

from lt_sdk.common import py_file_utils
from lt_sdk.graph.transform_graph import utils
from lt_sdk.verification import performance_sweep
from lt_sdk.verification.tools import compute_mAP, object_detection_utils

# Metrics/stats
TOP_K_FORMAT = "top_{}_acc"
ACCURACY = TOP_K_FORMAT.format(1)
MAP = "mAP"


class TopKAccuracyWorkload(performance_sweep.PerformanceSweep):

    def prediction_edge(self):
        """Returns the name of the prediction edge"""
        raise NotImplementedError()

    def k_list(self):
        """Returns a list of different k values to check"""
        raise NotImplementedError()

    def get_top_k_predictions(self, predictions, k):
        """Returns the top k predictions, can override to compute this differently"""
        assert (predictions.shape[1] >= k)
        return predictions.argsort(axis=1)[:, -k:]

    def init_new_config(self):
        self._correct = {k: 0 for k in self.k_list()}
        self._total = 0

    def update_quality_metrics(self, performance_data, test_outputs, labels):
        # Get the predictions
        predictions = []
        for inf_out in test_outputs.batches:
            for named_tensor in inf_out.results:
                if named_tensor.edge_info.name.startswith(self.prediction_edge()):
                    predictions.append(
                        utils.tensor_pb_to_array(named_tensor.data,
                                                 np.float32))
        predictions = np.concatenate(predictions, axis=0)

        # Format the arrays
        num_samples = min(labels.shape[0], predictions.shape[0])
        predictions = predictions[:num_samples]
        predictions = predictions.reshape(num_samples, -1)
        labels = labels[:num_samples]
        labels = labels.reshape(num_samples, 1)

        # Calculate top k accuracy for each value of k
        self._total += num_samples
        for k in self.k_list():
            self._correct[k] += np.sum(
                labels == self.get_top_k_predictions(predictions,
                                                     k))
            performance_data.quality_metrics.metrics[TOP_K_FORMAT.format(
                k)] = self._correct[k] / self._total


class AccuracyWorkload(TopKAccuracyWorkload):

    def output_is_argmax(self):
        """Returns true if the output is argmax instead of softmax"""
        return False

    def get_top_k_predictions(self, predictions, k):
        if self.output_is_argmax():
            assert (k == 1)
            assert (predictions.shape[1] == 1)
            return predictions

        return super().get_top_k_predictions(predictions, k)

    def k_list(self):
        return [1]


class Top5AccuracyWorkload(TopKAccuracyWorkload):

    def k_list(self):
        return [1, 5]


class BoundingBox(object):

    def __init__(self, det_cls, score, box):
        """
        params:
            det_cls: The class of the object detected.
            score: Confidence level.
            box: A tuple or list of 4 number indicating the coordinates of
                upper-left and bottom-right corners.  (is this true?)
        """
        self._det_cls = det_cls
        self._score = score
        self._box = box

    def format_to_string(self):
        return "{0} {1} {2} {3} {4} {5}\n".format(self._det_cls,
                                                  self._score,
                                                  self._box[0],
                                                  self._box[1],
                                                  self._box[2],
                                                  self._box[3])


class ObjDetResult(object):

    def __init__(self, img_index):
        self._boxes = []
        self._img_index = img_index

    def add_box(self, box):
        """Adds a BoundingBox object to this collection of results."""
        self._boxes.append(box)

    def write_result(self, write_dir):
        label_name = "{}.txt".format(self._img_index)
        result_str = "".join([x.format_to_string() for x in self._boxes])

        with open(os.path.join(write_dir, label_name), "a") as f:
            f.write(result_str)


class ObjectDetectionWorkload(performance_sweep.PerformanceSweep):

    def original_image_size(self):
        """Return a list of [image height, image width]."""
        raise NotImplementedError()

    def prediction_edge(self):
        """Returns the name of the prediction edge"""
        raise NotImplementedError()

    def init_new_config(self):
        # Need to write ground truth and detection results to disk to compute mAP
        self._ground_truth_dir = py_file_utils.mkdtemp()
        self._detection_results_dir = py_file_utils.mkdtemp()
        self._image_indx = 0  # tracks image index over stream of test outputs

    def end_of_config(self):
        shutil.rmtree(self._ground_truth_dir)
        shutil.rmtree(self._detection_results_dir)

    def inference_to_obj_det_results(self, inf_out, test_images_sizes, start_img):
        results = []
        for named_tensor in inf_out.results:
            if named_tensor.edge_info.name.startswith(self.prediction_edge()):
                raw = utils.tensor_pb_to_array(named_tensor.data, np.float32)
                logging.info(raw.shape)
                for j in range(raw.shape[0]):
                    label_ind = start_img + j

                    # Post-processing to convert output boxes to detection results
                    boxes = object_detection_utils.non_max_suppression(raw[j:j + 1], 0.5)
                    detection_result = ObjDetResult(label_ind)

                    for det_cls, bboxs in boxes.items():
                        for box, score in bboxs:
                            new_box = object_detection_utils.\
                                convert_to_original_size(
                                box, np.array(self.original_image_size()),
                                test_images_sizes[label_ind], True
                            )

                            detection_result.add_box(BoundingBox(
                                det_cls,
                                score,
                                new_box))

                    results.append(detection_result)

                continue
        return results

    def update_quality_metrics(self, performance_data, test_outputs, labels):
        # Labels and image sizes
        test_labels_dir, test_images_sizes = labels

        assert (len(test_outputs.batches) > 0)
        assert (len(test_outputs.batches[0].results) > 0)
        img_ind = 0
        # Assumes dim0 of the first thing in results has the batch dim.
        for batch in test_outputs.batches:
            for j in range(batch.results[0].edge_info.shape.d[0]):
                # Copy test label to ground truth directory
                # NOTE: assumes graph_output[i] corresponds to
                # label "{}.txt".format(self._image_indx)
                label_name = "{}.txt".format(img_ind)
                shutil.copy2(os.path.join(test_labels_dir,
                                          label_name),
                             self._ground_truth_dir)
                img_ind += 1

        all_results = []
        for inf_out in test_outputs.batches:
            these = self.inference_to_obj_det_results(inf_out,
                                                      test_images_sizes,
                                                      self._image_indx)
            all_results.extend(these)
            self._image_indx += len(these)

        for res in all_results:
            res.write_result(self._detection_results_dir)

        # Compute the mAP
        mAP = compute_mAP.main(self._ground_truth_dir, self._detection_results_dir)
        performance_data.quality_metrics.metrics[MAP] = mAP
