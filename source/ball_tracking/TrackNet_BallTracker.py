from source.ball_tracking.TrackNet3 import TrackNet3
from utils import get_video_properties

import torch
import numpy as np
import cv2
import time
import os

BATCH_SIZE = 1
HEIGHT = 288
WIDTH = 512

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU Use : ', torch.cuda.is_available())


def custom_time(time):
    remain = int(time / 1000)
    ms = (time / 1000) - remain
    s = remain % 60
    s += ms
    remain = int(remain / 60)
    m = remain % 60
    remain = int(remain / 60)
    h = remain
    # Generate custom time string
    cts = ''
    if len(str(h)) >= 2:
        cts += str(h)
    else:
        for i in range(2 - len(str(h))):
            cts += '0'
        cts += str(h)

    cts += ':'

    if len(str(m)) >= 2:
        cts += str(m)
    else:
        for i in range(2 - len(str(m))):
            cts += '0'
        cts += str(m)

    cts += ':'

    if len(str(int(s))) == 1:
        cts += '0'
    cts += str(s)

    return cts


class BallTracking:

    def __init__(self, video_name, Cfg, debug=False):
        # empty Torch cache memory
        torch.cuda.empty_cache()

        self.video_name = video_name
        self.video_capture = None
        self.video_writer = None
        self.v_height = None
        self.v_width = None
        self.v_num_frames = None
        self.v_fps = None
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.ratio_w = None
        self.ratio_h = None

        self.checkpoint = None
        self.learning_rate = 0.1
        self.optimizer = 'Ada'
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.seed = 1

        self.device = device
        self.model = None
        self.model_HEIGHT = 288
        self.model_WIDTH = 512

        self.output_csv_file = None
        self.count = 0
        self.count2 = -3
        self.time_list = []
        self.rets = []
        self.images = []
        self.frame_times = []

        self.min_h_pred = 0.5
        self.debug = debug
        self.save_path = None
        self.input_path = None
        self.code_config = Cfg
        self.weights_path = os.path.join(self.code_config.get_TRACKNET_CHECKPOINT_DIR(), 'TrackNet3_30.tar')
        self.load(Cfg=Cfg)

    def load(self, Cfg):

        # Define device to be used in prediction.
        self.device = device
        self.model = TrackNet3()
        self.model.to(device)

        # Define video name in class and related paths
        self.input_path = os.path.join(Cfg.get_INPUT_VIDEO_DIR(), self.video_name)
        name = self.video_name.split('.mp4')[0]
        self.save_path = os.path.join(Cfg.get_RESULT_VIDEO_OUTPUT(), name + '_ball_tracking.mp4')

        # Read input video
        video = os.path.join(Cfg.get_INPUT_VIDEO_DIR(), self.video_name)
        self.video_name = video
        self.video_capture = cv2.VideoCapture(self.video_name)

        self.v_fps, self.v_num_frames, self.v_width, self.v_height = get_video_properties(video=self.video_capture)

        self.ratio_w = self.v_width / WIDTH
        self.ratio_h = self.v_height / HEIGHT

        # define codec and create VideoWriter object
        self.video_writer = cv2.VideoWriter(self.save_path, self.fourcc, self.v_fps,
                                            (self.v_width, self.v_height), True)

        # Create .csv file to store predictions.
        result_path = os.path.join(Cfg.get_RESULT_VIDEO_OUTPUT(), 'ball_tracking_predict.csv')
        self.output_csv_file = open(result_path, 'w')
        self.output_csv_file.write('Frame,Visibility,X,Y,Time\n')

    def track_ball(self):

        if self.optimizer == 'Ada':
            self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.learning_rate,
                                                  rho=0.9, eps=1e-06, weight_decay=0)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                             weight_decay=self.weight_decay, momentum=self.momentum)

        self.checkpoint = torch.load(self.weights_path)
        self.model.load_state_dict(self.checkpoint['state_dict'])

        self.model.eval()
        start1 = time.time()
        while True:
            # Clean lists for every iteration.
            self.rets = []
            self.images = []
            self.frame_times = []

            for idx in range(3):
                # Read frame from webcam
                ret, frame = self.video_capture.read()
                t = custom_time(self.video_capture.get(cv2.CAP_PROP_POS_MSEC))
                self.rets.append(ret)
                self.images.append(frame)
                self.frame_times.append(t)
                self.count += 1
                self.count2 += 1

            grays = []
            if all(self.rets):
                for img in self.images:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    grays.append(img[:, :, 0])
                    grays.append(img[:, :, 1])
                    grays.append(img[:, :, 2])
            elif self.count >= self.count:
                break
            else:
                print("read frame error. skip...")
                continue

            # TackNet prediction
            unit = np.stack(grays, axis=2)
            unit = cv2.resize(unit, (WIDTH, HEIGHT))
            unit = np.moveaxis(unit, -1, 0).astype('float32') / 255
            unit = torch.from_numpy(np.asarray([unit])).to(device)
            with torch.no_grad():
                start = time.time()
                h_pred = self.model(unit)
                end = time.time()
                self.time_list.append(end - start)
            h_pred = h_pred > 0.5
            h_pred = h_pred.cpu().numpy()
            h_pred = h_pred.astype('uint8')
            h_pred = h_pred[0] * 255

            for idx_f, (image, frame_time) in enumerate(zip(self.images, self.frame_times)):
                show = np.copy(image)
                show = cv2.resize(show, (frame.shape[1], frame.shape[0]))
                # Ball tracking
                if np.amax(h_pred[idx_f]) <= 0:  # no ball
                    self.output_csv_file.write(str(self.count2 + idx_f) + ',0,0,0,' + frame_time + '\n')
                    self.video_writer.write(image)
                else:
                    (cnts, _) = cv2.findContours(h_pred[idx_f].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    rects = [cv2.boundingRect(ctr) for ctr in cnts]
                    max_area_idx = 0
                    max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                    for i in range(len(rects)):
                        area = rects[i][2] * rects[i][3]
                        if area > max_area:
                            max_area_idx = i
                            max_area = area
                    target = rects[max_area_idx]
                    (cx_pred, cy_pred) = (int(self.ratio_w * (target[0] + target[2] / 2)),
                                          int(self.ratio_h * (target[1] + target[3] / 2)))

                    self.output_csv_file.write(str(self.count2 + idx_f) + ',1,' + str(cx_pred) +
                                               ',' + str(cy_pred) + ',' + frame_time + '\n')
                    cv2.circle(image, (cx_pred, cy_pred), 5, (0, 0, 255), -1)
                    self.video_writer.write(image)

        # Release all video components and files that store information.
        self.release_files(start1=start1)

    def release_files(self, start1):
        self.output_csv_file.close()
        self.video_capture.release()
        self.video_capture.release()
        end1 = time.time()
        print('Prediction time:', (end1 - start1), 'secs')
        print('FPS', self.v_num_frames / (end1 - start1))
        print('Done......')
