from lt_sdk.verification.workloads import (
    coco_sweep,
    imagenet_sweep,
    mnist_sweep,
    mrpc_sweep,
    resnet_cifar_sweep,
)

DATA_DIR = "external/training_data"

MNIST_SMALL_DNN = "mnist_small_dnn"
MNIST_SMALL_DNN_TF_2 = "mnist_small_dnn_tf_2"
MNIST_CNN = "mnist_cnn"
MNIST_CNN_FULL = "mnist_cnn_full"
CIFAR_RESNET_8 = "cifar_resnet_8"
CIFAR_RESNET_50 = "cifar_resnet_50"
CIFAR_RESNET_50_SPARSE = "cifar_resnet_50_sparse"
YOLOV3 = "yolo"
YOLOV3_TINY = "yolo_tiny"
IMGNET_RESNET_50 = "imgnet_resnet_50"
IMGNET_RESNET_50_FULL = "imgnet_resnet_50_full"
MOBILENET_V2 = "mobilenet"
FRCNN_RESNET50 = "faster_rcnn_resnet_50"
SSD_RESNET50 = "ssd_resnet50"
EFFICIENTNET_B0 = "efficientnet_b0"
MRPC_BERT = "mrpc_bert"

PERFORMANCE_SWEEP_MAP = {
    MNIST_SMALL_DNN: mnist_sweep.MnistSmallDNNSweep,
    MNIST_SMALL_DNN_TF_2: mnist_sweep.MnistSmallDNNTF2Sweep,
    MNIST_CNN: mnist_sweep.MnistCNNSweep,
    MNIST_CNN_FULL: mnist_sweep.FullMnistCNNSweep,
    CIFAR_RESNET_8: resnet_cifar_sweep.ResNet8CIFARSweep,
    CIFAR_RESNET_50: resnet_cifar_sweep.ResNet50CIFARSweep,
    CIFAR_RESNET_50_SPARSE: resnet_cifar_sweep.ResNet50SparseCIFARSweep,
    YOLOV3: coco_sweep.YOLOV3Sweep,
    YOLOV3_TINY: coco_sweep.YOLOV3TinySweep,
    IMGNET_RESNET_50: imagenet_sweep.ResNet50Sweep,
    IMGNET_RESNET_50_FULL: imagenet_sweep.FullResnet50Sweep,
    MOBILENET_V2: imagenet_sweep.MobileNetSweep,
    FRCNN_RESNET50: coco_sweep.FasterRCNNResNet50Sweep,
    SSD_RESNET50: coco_sweep.SSDResNet50Sweep,
    EFFICIENTNET_B0: imagenet_sweep.EfficientNetSweep,
    MRPC_BERT: mrpc_sweep.BERTMRPCSweep,
}


def get_sweep(workload, out_dir, **kwargs):
    return PERFORMANCE_SWEEP_MAP[workload](out_dir, default_data_dir=DATA_DIR, **kwargs)
