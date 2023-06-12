# import some common libraries
import os.path
import torchvision

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from config import cfg as Cfg

logger = setup_logger()
detectron_config = get_cfg()
code_config = Cfg()


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
