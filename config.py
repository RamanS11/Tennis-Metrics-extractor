import os


class cfg:

    # Define root.
    # route = '/home/tfg_23'
    def __init__(self):
        self.FINAL_DIR = None
        self.route = os.getcwd()

        # Define Data folder and it's children.
        self.DATA_DIR = os.path.join(self.route, 'data')
        self.CODE_DIR = os.path.join(self.route, 'source')

        self.INPUT_VIDEO_DIR = os.path.join(self.DATA_DIR, 'VideoInput')
        self.OUTPUT_VIDEO_DIR = os.path.join(self.DATA_DIR, 'VideoOutput')

        # Indicates path where to store results generated in pose estimation and ball tracking!
        self.RESULT_VIDEO_OUTPUT = None

        # Define Court directories
        self.COURT_CODE_DIR = os.path.join(self.CODE_DIR, 'court_detection')
        self.COURT_DATA_DIR = os.path.join(self.DATA_DIR, 'courtConfig')
        self.COURT_DATA_CONFIG = os.path.join(self.COURT_DATA_DIR, 'configuration')
        self.COURT_DATA_RESULT = os.path.join(self.COURT_DATA_DIR, 'results')
        self.COURT_DATA_GT = os.path.join(self.COURT_DATA_DIR, 'groundTruth')

        # Define Pose directories
        self.POSE_CODE_DIR = os.path.join(self.CODE_DIR, 'pose_estimation')
        self.POSE_DATA_DIR = os.path.join(self.DATA_DIR, 'poseConfig')
        self.POSE_DATA_WEIGHTS = os.path.join(self.POSE_DATA_DIR, 'keypointRCNN')
        self.KEYPOINTS_WEIGHTS = os.path.join(self.POSE_DATA_WEIGHTS, 'keypointrcnn_resnet50_fpn_coco-9f466800.pth')
        self.KEYPOINTS_WEIGHTS_LEGACY = os.path.join(self.POSE_DATA_WEIGHTS,
                                                     'keypointrcnn_resnet50_fpn_coco-fc266e95.pth')

        # Define Ball directories
        self.BALL_SOURCE_DIR = os.path.join(self.CODE_DIR, 'ball_tracking')
        self.BALL_DATA_DIR = os.path.join(self.DATA_DIR, 'ballConfig')

        if not os.path.exists(self.COURT_DATA_RESULT):
            os.mkdir(self.COURT_DATA_RESULT)

        # Define different data paths.
        self.COURT_DATA_REF = os.path.join(self.DATA_DIR, "court_config")

    def get_INPUT_VIDEO_DIR(self):
        return self.INPUT_VIDEO_DIR

    def get_keypointRCNN_LEGACY_PATH(self):
        return self.KEYPOINTS_WEIGHTS_LEGACY

    def set_destination_directory(self, path):
        path = path.split('.mp4')[0]
        self.FINAL_DIR = os.path.join(self.COURT_DATA_RESULT, path)

    def get_COURT_GT_DIR(self):
        return self.COURT_DATA_GT

    def get_COURT_CONF_DIR(self):
        return self.COURT_DATA_CONFIG

    def get_COURT_RESULTS_DIR(self):
        return self.COURT_DATA_RESULT

    def get_DATA_DIR(self):
        return self.DATA_DIR

    def get_TRACKNET_CHECKPOINT_DIR(self):
        return self.BALL_DATA_DIR

    def get_FINAL_DIR(self):
        if self.FINAL_DIR is not None:
            return self.FINAL_DIR
        else:
            print('final directory not defined!')
            print('Note that results are being stored at VideoOutput directory')
            return self.OUTPUT_VIDEO_DIR

    def set_result_directory(self, video_name):
        video_name = video_name.split('.mp4')[0]
        self.RESULT_VIDEO_OUTPUT = os.path.join(self.OUTPUT_VIDEO_DIR, video_name)
        if not os.path.exists(self.RESULT_VIDEO_OUTPUT):
            os.mkdir(self.RESULT_VIDEO_OUTPUT)
            print('Created folder: ', video_name, ' in /data/VideoOutput directory')
        else:
            print('Folder already exist, files will be overwritten')

    def get_RESULT_VIDEO_OUTPUT(self):

        if not self.RESULT_VIDEO_OUTPUT:
            print('No output folder created to store results for this video!')
            return self.OUTPUT_VIDEO_DIR
        else:
            return self.RESULT_VIDEO_OUTPUT
