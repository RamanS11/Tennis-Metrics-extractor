# import some common libraries
import os.path
import torchvision

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from config import cfg as Cfg

import torchvision
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator

logger = setup_logger()
detectron_config = get_cfg()
code_config = Cfg()


def load_model():
    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    # KeypointRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be ['0']. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)
    # put the pieces together inside a KeypointRCNN model
    model = KeypointRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,
                         keypoint_roi_pool=keypoint_roi_pooler)
    model.eval()
    model.eval()

    return model


def KeypointPredictor():
    """
    Method that will construct and return Default KeypointRCNN Predictor.
    :return: pose_detector (Detectron2's Default predictor using KeypointRCNN model)
    """

    # Define models config
    detectron_config.MODEL.DEVICE = "cuda"
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    model.eval().cuda()
    detectron_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
    detectron_config.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    detectron_config.MODEL.WEIGHTS = os.path.join(code_config.get_keypointRCNN_LEGACY_PATH())

    # Define how many players to be detected in each frame:
    detectron_config.TEST.DETECTIONS_PER_IMAGE = 2

    # Build model and charge predictor
    model = build_model(detectron_config)
    predictor = DefaultPredictor(detectron_config)

    return predictor
