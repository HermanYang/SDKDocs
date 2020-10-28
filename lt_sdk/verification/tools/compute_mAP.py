import argparse
import glob
import json
import logging
import math
import os
import shutil
import sys

import numpy as np

from lt_sdk.common import py_file_utils

MINOVERLAP = 0.5  # default value (defined in the PASCAL VOC2012 challenge)


def log_average_miss_rate(precision, fp_cumsum, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.

        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    # if there were no detections of that class
    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index
        # since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi


def error(msg):
    print(msg)
    sys.exit(0)


def is_float_between_0_and_1(value):
    try:
        val = float(value)
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]

    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def main(ground_truth_dir, detection_results_dir, ignore=None, set_class_iou=None):
    # Defaults from from args
    GT_PATH = ground_truth_dir
    DR_PATH = detection_results_dir
    specific_iou_flagged = False
    if set_class_iou is not None:
        specific_iou_flagged = True
    if ignore is None:
        ignore = []

    # Create a temp dir
    TEMP_FILES_PATH = py_file_utils.mkdtemp()

    # get a list with the ground-truth files
    ground_truth_files_list = glob.glob(GT_PATH + "/*.txt")
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}

    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        # check if there is a correspondent detection-results file
        temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error_msg += "(You can avoid this error message by running"
            error_msg += "extra/intersect-gt-and-dr.py)"
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)
        # create ground-truth dictionary
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
                else:
                    class_name, left, top, right, bottom = line.split()
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <left> <top> <right> <bottom>"
                error_msg += "[\"difficult\"]\n"
                error_msg += " Received: " + line
                error_msg += "\n\nIf you have a <class_name> with spaces "
                error_msg += "between words you should remove them\n"
                error_msg += "by running the script \"remove_space.py\" or "
                error_msg += "rename_class.py\" in the \"extra/\" folder."
                error(error_msg)
            # check if class is in the ignore list, if yes skip
            if class_name in ignore:
                continue
            bbox = left + " " + top + " " + right + " " + bottom
            if is_difficult:
                bounding_boxes.append({
                    "class_name": class_name,
                    "bbox": bbox,
                    "used": False,
                    "difficult": True
                })
                is_difficult = False
            else:
                bounding_boxes.append({
                    "class_name": class_name,
                    "bbox": bbox,
                    "used": False
                })
                # count that object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if class didn"t exist yet
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        # if class didn"t exist yet
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

        # dump bounding_boxes into a ".json" file
        with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json",
                  "w") as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    # let"s sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    if specific_iou_flagged:
        n_args = len(set_class_iou)
        error_msg = \
            "\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]"
        if n_args % 2 != 0:
            error("Error, missing arguments. Flag usage:" + error_msg)
        # [class_1] [IoU_1] [class_2] [IoU_2]
        # specific_iou_classes = ["class_1", "class_2"]
        specific_iou_classes = set_class_iou[::2]  # even
        # iou_list = ["IoU_1", "IoU_2"]
        iou_list = set_class_iou[1::2]  # odd
        if len(specific_iou_classes) != len(iou_list):
            error("Error, missing arguments. Flag usage:" + error_msg)
        for tmp_class in specific_iou_classes:
            if tmp_class not in gt_classes:
                error("Error, unknown class \"" + tmp_class + "\". Flag usage:" +
                      error_msg)
        for num in iou_list:
            if not is_float_between_0_and_1(num):
                error("Error, IoU must be between 0.0 and 1.0. Flag usage:" + error_msg)

    # get a list with the detection-results files
    dr_files_list = glob.glob(DR_PATH + "/*.txt")
    dr_files_list.sort()

    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            # the first time it checks if all the corresponding ground-truth files exist
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error_msg += "(You can avoid this error message "
                    error_msg += "by running extra/intersect-gt-and-dr.py)"
                    error(error_msg)
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file
                    error_msg += " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left>"
                    error_msg += " <top> <right> <bottom>\n"
                    error_msg += " Received: " + line
                    error(error_msg)
                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append({
                        "confidence": confidence,
                        "file_id": file_id,
                        "bbox": bbox
                    })

        # sort detection-results by decreasing confidence
        bounding_boxes.sort(key=lambda x: float(x["confidence"]), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", "w") as outfile:
            json.dump(bounding_boxes, outfile)

    # Calculate the AP for each class
    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    count_true_positives = {}
    for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0
        # Load detection-results of that class
        dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
        with open(dr_file) as f:
            dr_data = json.load(f)

        # Assign detection-results to ground-truth objects
        nd = len(dr_data)
        tp = [0] * nd  # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]

            # assign detection-results to ground truth object if any
            # open ground-truth with that file_id
            gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
            with open(gt_file) as f:
                ground_truth_data = json.load(f)
            ovmax = -1
            gt_match = -1
            # load detected object bounding-box
            bb = [float(x) for x in detection["bbox"].split()]
            for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name:
                    bbgt = [float(x) for x in obj["bbox"].split()]
                    bi = [
                        max(bb[0],
                            bbgt[0]),
                        max(bb[1],
                            bbgt[1]),
                        min(bb[2],
                            bbgt[2]),
                        min(bb[3],
                            bbgt[3])
                    ]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = ((bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) +
                              (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) -
                              iw * ih)
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

            # assign detection as true positive/don"t care/false positive
            # set minimum overlap
            min_overlap = MINOVERLAP
            if specific_iou_flagged:
                if class_name in specific_iou_classes:
                    index = specific_iou_classes.index(class_name)
                    min_overlap = float(iou_list[index])
            if ovmax >= min_overlap:
                if "difficult" not in gt_match:
                    if not bool(gt_match["used"]):
                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                        # update the ".json" file
                        with open(gt_file, "w") as f:
                            f.write(json.dumps(ground_truth_data))
                    else:
                        # false positive (multiple detection)
                        fp[idx] = 1
            else:
                # false positive
                fp[idx] = 1

        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val

        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]

        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap
        ap_dictionary[class_name] = ap

        n_images = counter_images_per_class[class_name]
        lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
        lamr_dictionary[class_name] = lamr

    mAP = sum_AP / n_classes
    text = "mAP = {0:.2f}%".format(mAP * 100)
    logging.info(text)

    # remove the temp_files directory
    shutil.rmtree(TEMP_FILES_PATH)

    return mAP


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # argparse receiving list of classes to be ignored
    parser.add_argument("-i",
                        "--ignore",
                        nargs="+",
                        type=str,
                        help="ignore a list of classes.")
    # argparse receiving list of classes with specific IoU
    # (e.g., python main.py --set-class-iou person 0.7)
    parser.add_argument("--set-class-iou",
                        nargs="+",
                        type=str,
                        help="set IoU for a specific class.")

    parser.add_argument("--ground_truth_dir",
                        type=str,
                        help="directory where ground truth files are")
    parser.add_argument("--detection_results_dir",
                        type=str,
                        help="directory where detection results are")

    args = parser.parse_args()

    main(args.ground_truth_dir,
         args.detection_results_dir,
         ignore=args.ignore,
         set_class_iou=args.set_class_iou)
