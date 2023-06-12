import os

import cv2
import numpy as np
import torch
import torchvision.models as models

from PIL import Image
from torchvision.transforms import transforms as transforms
from skimage.feature import local_binary_pattern

import source.pose_estimation.utils as utils_pose
from source.pose_estimation.sort import Sort

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU Use : ', torch.cuda.is_available())


def get_model(min_size=800):
    # Initialize the model
    model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True, num_keypoints=17, min_size=min_size)

    return model


def bbox_center(bbox):
    x0, y0, x1, y1 = bbox

    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    return center_x, center_y


def candidate_dist_score(box, line_center, score):
    center_x, center_y = bbox_center(box)
    distance = np.sqrt((line_center[0] - center_x) ** 2 + (line_center[1] - center_y) ** 2)

    return distance


def distance_point_to_line(bbox, baseline_start, baseline_end):
    """
    Compute distance between candidate bounding box and any point in (top) baseline, to detect player in first frame.
    :param bbox: Candidate bounding box in (x0, y0, x1, y1) format.
    :param baseline_start: top baseline starting point.
    :param baseline_end: top baseline ending point.
    :return: perpendicular distance between candidate bounding box center and top baseline.
    """
    # Compute center of candidate bounding box.
    point = bbox_center(bbox=bbox)
    point = np.array(point)

    line_start = np.array(baseline_start)
    line_end = np.array(baseline_end)

    # Compute the vector representing the line segment AB and the line segment AC
    AB = line_end - line_start
    AC = point - line_start

    # Compute the dot product of AB and AC, to get the squared length of AB
    dot_product = np.dot(AB, AC)
    length_squared = np.dot(AB, AB)
    # Compute the parameter t
    t = dot_product / length_squared
    # Calculate the closest point on the line to point C, P
    P = line_start + t * AB

    # Calculate the distance between point C and point P
    distance = np.linalg.norm(point - P)

    return distance


def distance_to_prev_detection(last_bbox, candidate_bbox):
    """
    Compute Euclidean distance between center of last bounding box and candidate bounding box.
    :param last_bbox: last detected bounding box
    :param candidate_bbox: candidate bounding box
    :return: Euclidean Distance between center points.
    """
    l_center_x, l_center_y = bbox_center(bbox=last_bbox)
    c_center_x, c_center_y = bbox_center(bbox=candidate_bbox)
    dist = np.linalg.norm(np.array(l_center_x, l_center_y) - np.array(c_center_x, c_center_y))
    return dist


def boxes_dist(boxes):
    """
    Calculate the cumulative distance of all the boxes
    """
    total_dist = 0
    for box1, box2 in zip(boxes, boxes[1:]):
        box1_center = np.array(bbox_center(bbox=box1))
        box2_center = np.array(bbox_center(bbox=box2))
        dist = np.linalg.norm(box2_center - box1_center)
        total_dist += dist
    return total_dist


def find_closest_index(arr, current):
    distances = np.linalg.norm(arr - current, axis=1)
    index = np.argmin(distances)
    return index


def calculate_bhattacharyya_distance(hist1, hist2):
    # Normalize histograms
    norm_hist1 = hist1 / np.sum(hist1)
    norm_hist2 = hist2 / np.sum(hist2)

    # Calculate Bhattacharyya coefficient
    b_coefficient = np.sum(np.sqrt(norm_hist1 * norm_hist2))

    # Calculate Bhattacharyya distance
    distance = np.sqrt(1 - b_coefficient)

    return 1 - distance


def area_of_box(box):
    height = box[3] - box[1]
    width = box[2] - box[0]
    return height * width


class playersDetection:

    def __init__(self, video_name, Cfg, debug=True):

        # empty Torch cache memory
        torch.cuda.empty_cache()
        # Define Video related class variables
        self.video_name = None
        self.input_path = None
        self.save_path = None
        self.height = None
        self.width = None
        self.frame_count = None
        self.total_fps = None
        self.frame = None
        self.writer = None
        self.capture = None

        # Define model related class variables.
        self.model = None
        self.device = None
        self.min_size = None
        self.transform = None

        # Define bottom player detection and tracking related class variables
        self.player_bottom_detections = []
        self.player_bottom_keypoints = {}
        self.player_bottom_boxes = {}
        self.bottom_misses = 0

        # Define top player detection related class variables
        self.sort_MOT = Sort(max_age=10, min_hits=3, iou_threshold=0.05)
        self.candidate_first_appearance = {}
        self.top_candidate_appearance = {}
        self.top_candidate_detections = {}
        self.top_candidate_keypoints = {}
        self.top_candidate_boxes = {}
        self.top_player_first_h = None
        self.top_misses = 0
        self.min_top_thr = 0.5

        # Define top player tracking related class variables
        self.best_top_candidate_id = None
        self.top_player_IDs = []
        self.top_player_initial_lbp = None
        self.top_player_last_lbp = None
        self.top_candidate_dists = {}
        self.player_top_keypoints = {}
        self.player_top_boxes = {}

        # Define first detections related class variables.
        self.top_player_margin = 50
        self.bottom_player_margin = 100
        self.min_distance_to_baseline = 150

        self.top_baseline = None
        self.bottom_baseline = None

        self.debug = debug
        self.load(video_name=video_name, Cfg=Cfg)

    def load(self, video_name, Cfg, min_size=775):

        # Define device to be used in prediction.
        self.device = device
        self.min_size = min_size

        # initialize all parameters needed to compute pose pose estimation and model
        self.init_transforms()
        self.init_model()

        # Define video name in class and related paths
        self.video_name = video_name
        self.input_path = os.path.join(Cfg.get_INPUT_VIDEO_DIR(), self.video_name)
        self.save_path = os.path.join(Cfg.get_RESULT_VIDEO_OUTPUT(), self.video_name)

        # Read input video
        video = os.path.join(Cfg.get_INPUT_VIDEO_DIR(), video_name)
        self.video_name = video
        vid = cv2.VideoCapture(self.video_name)
        return_value, frame = vid.read()

        if self.debug:
            if return_value:
                self.total_fps = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                self.top_player_IDs = [None] * self.total_fps
                self.width, self.height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                    vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                self.width, self.height = 1280, 720
                print('Not able to initialize video!, check video name.')
                return

            # define codec and create VideoWriter object
            self.writer = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                          vid.get(cv2.CAP_PROP_FPS), (self.width, self.height), True)

        vid.release()

    def get_video_writer(self):
        return self.writer

    def release_video(self):
        # release video capturer
        self.writer.release()

        # close all frames and video windows
        cv2.destroyAllWindows()

    def init_transforms(self):
        # transform to convert the image to tensor
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def init_model(self):
        # load the model on the computation device
        self.model = get_model(min_size=self.min_size).to(device=self.device).eval()

    def detect_candidates_top(self, frame, top_mask, baseline, frame_idx):

        boxes = frame.copy()
        iteration_mask = top_mask

        self.top_baseline = baseline

        # Detect all persons in the top part of the court.
        outputs = self.detect(orig_frame=top_mask, image_court=iteration_mask)
        num_detections = len(outputs[0]['boxes'])

        if num_detections > 0:

            # Filter detections with score < self.min_top_thr
            scores = outputs[0]['scores'][:].cpu().detach().numpy()
            index = np.where(scores > self.min_top_thr)[0]

            # Store all detections whose detection score > self.min_top_thr
            if len(index) > 0:
                scores = outputs[0]['scores'][outputs[0]['scores'] > self.min_top_thr].cpu().detach().numpy()
                bboxes = outputs[0]['boxes'][outputs[0]['scores'] > self.min_top_thr].cpu().detach().numpy()

                # If we are in the first frame, add restriction of distance to baseline.
                # For the next detections, Kalman filter is applied in SORT tracker.
                if frame_idx == 0 or self.best_top_candidate_id is None:
                    bl_x1, bl_y1, bl_x2, bl_y2 = baseline
                    distances = [distance_point_to_line(bbox=bboxes[j], baseline_start=(bl_x1, bl_y1),
                                                        baseline_end=(bl_x2, bl_y2)) for j in index]
                    scores = (1 / np.array(distances)) * 0.65 + scores * 0.45
                    outputs_id = np.argmax(scores)
                    print('Candidate id: ', outputs_id, ' distance: ', distances, ' and score: ', scores)
                    if distances[outputs_id] < 100.:
                        self.best_top_candidate_id = outputs_id + 1
                        self.top_player_first_h = self.model_top_candidate(image=frame, bbox=bboxes[outputs_id])
                    else:
                        bboxes = None
                        scores = None
            else:
                bboxes = None
                scores = None

            # Start tracking with SORT
            tracked_candidates = self.sort_MOT.update(bboxes, scores)

            # Store information for each candidate in top court.
            for box in tracked_candidates:

                candidate_id = int(box[4])
                hist = self.model_top_candidate(image=frame, bbox=box[:4])

                # If current candidate never appeared, indicate as new detection and create space in dictionaries.
                if candidate_id not in self.top_candidate_boxes.keys():
                    self.top_candidate_boxes[candidate_id] = ([None] * self.total_fps)
                    self.top_candidate_appearance[candidate_id] = ([None] * self.total_fps)
                    self.top_candidate_keypoints[candidate_id] = ([None] * self.total_fps)
                    self.candidate_first_appearance[candidate_id] = frame_idx

                # Find keypoints detected for current element tracked to be stored in dictionary of specific id.
                current_bbox = box[:4]
                detection_index = find_closest_index(arr=bboxes, current=current_bbox)
                keypoints = outputs[0]['keypoints'][detection_index].cpu().detach().numpy()

                self.top_candidate_boxes[candidate_id][frame_idx] = box[:4]
                self.top_candidate_keypoints[candidate_id][frame_idx] = keypoints
                self.top_candidate_appearance[candidate_id][frame_idx] = hist

                cv2.rectangle(boxes, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 0], 2)
                cv2.putText(boxes, f'Player {int(box[4])}', (int(box[0]) - 10, int(box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        self.top_player_IDs[frame_idx] = self.best_top_candidate_id

        return boxes

    def model_top_candidate(self, image, bbox):
        """
        Having all the detections in the top of the court, select the best candidate by using minimum distance from
        bounding box center to top baseline.
        Once the best candidate is defined, codify texture information of bounding box by means of Local Binary Patterns
        (LBP descriptor) in order to compare it all over the video and refine the tracking.
        # Note that best candidate bounding box: self.top_candidate_boxes[self.best_top_candidate_id][0]
        """

        # Crop image using the best candidate's bbox as ROI.
        bt_x0, bt_y0, bt_x1, bt_y1 = bbox
        bt_x0, bt_y0, bt_x1, bt_y1 = max(int(bt_x0), 0), max(int(bt_y0), 0), \
            min(int(bt_x1), int(self.height)), min(int(bt_y1), int(self.width))
        roi = image[bt_y0:bt_y1, bt_x0:bt_x1, :]
        try:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        except:
            gray_roi = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute LBP descriptor from ROI, to help in ReID if player is lost.
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_roi, n_points, radius, method='uniform')

        # Extract and normalize histogram from LBP descriptors to define target regarding bbox size.
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)

        return hist

    def model_top_candidates_colorH(self, image, bbox):
        """
        Use color information to model candidate, in order to have a better tracking and Re-Identification.
        """

        # Crop image using the best candidate's bbox as ROI.
        bt_x0, bt_y0, bt_x1, bt_y1 = bbox
        bt_x0, bt_y0, bt_x1, bt_y1 = max(int(bt_x0), 0), max(int(bt_y0), 0), \
            min(int(bt_x1), int(self.height)), min(int(bt_y1), int(self.width))
        roi = image[bt_y0:bt_y1, bt_x0:bt_x1, :]

        # Compute color histogram for ROI
        bins = 8  # Define number of bins.
        hist = cv2.calcHist([roi], [0, 1, 2], None, [bins, bins, bins], [0, 128, 0, 128, 0, 128])
        hist = cv2.normalize(hist, hist).flatten()

        return hist

    def detect_player_bottom(self, frame, bottom_mask, baseline, frame_idx):

        # If not bottom player detected yet, look for candidates.
        if not self.player_bottom_detections or not any(val is True for val in self.player_bottom_detections):

            iteration_mask = bottom_mask
            self.bottom_baseline = baseline

            # Detect all persons in the ROI
            outputs = self.detect(orig_frame=frame, image_court=iteration_mask)
            # No previous detections, compute feature extraction to track over videos
            bottom_player = self.get_initial_detection(detections=outputs, bottom=True, baseline=self.bottom_baseline)

        else:
            # Get last detected bounding box for bottom player and define new searching windows.
            detections = [i for i, x in enumerate(self.player_bottom_detections) if x]
            last_detection = detections[-1]
            last_bbox = self.player_bottom_boxes[last_detection]

            iteration_mask = self.get_iteration_mask(last_box=last_bbox, mask=frame,
                                                     margin=self.top_player_margin)

            # Detect people in current frame with searching window set from last detection.
            outputs = self.detect(orig_frame=frame, image_court=iteration_mask)
            bottom_player = self.track_player(detections=outputs, last_det=last_bbox, bottom=True)
            if not bottom_player:
                self.player_bottom_detections.append(False)
                self.player_bottom_boxes[frame_idx] = self.player_bottom_boxes[last_detection]
                self.player_bottom_keypoints[frame_idx] = self.player_bottom_keypoints[last_detection]
                return

        # output_image = frame
        if bottom_player:
            player_keypoints = bottom_player[0]
            bbox = bottom_player[1]
            self.player_bottom_boxes[frame_idx] = bbox
            self.player_bottom_keypoints[frame_idx] = player_keypoints
            self.player_bottom_detections.append(True)
            # output_image = utils_pose.draw_keypoints_and_boxes(player_keypoints, bbox, frame)
        else:
            self.player_bottom_detections.append(False)

        # press `q`to exit
        if 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return

    def get_initial_detection(self, detections, baseline, bottom=True):

        num_detections = len(detections[0]['boxes'])
        if num_detections > 0:
            scores = detections[0]['scores'][:].cpu().detach().numpy()
            index = np.where(scores > 0.85)[0]
            candidate_scores = np.zeros((len(index), 1))

            if len(index) > 0:
                for i in range(len(index)):
                    bbox = detections[0]['boxes'][index[i]].cpu().detach().numpy()
                    score = detections[0]['scores'][index[i]].cpu().detach().numpy()

                    bl_x1, bl_y1, bl_x2, bl_y2 = baseline
                    distance = distance_point_to_line(bbox=bbox, baseline_start=(bl_x1, bl_y1),
                                                      baseline_end=(bl_x2, bl_y2))

                    baseline_score = (1 / distance) * 0.5 + score * 0.5

                    # If distance to detected player is grater than a certain threshold, do not consider candidate as
                    # possible first detection, and add score = 0.
                    if distance > self.min_distance_to_baseline:
                        print(baseline_score)
                        candidate_scores[i, :] = 0
                    else:
                        candidate_scores[i, :] = baseline_score

                best_candidate = np.argmax(candidate_scores)
                bbox = detections[0]['boxes'][best_candidate].cpu().detach().numpy()
                keypoints = detections[0]['keypoints'][best_candidate].cpu().detach().numpy()
                return [keypoints, bbox]
            else:
                print('Bottom player candidate detected, but with small confidence!')
                if bottom:
                    self.bottom_misses += 1
                else:
                    self.top_misses += 1
                return None
        else:
            print('No bottom player detected!')
            if bottom:
                self.bottom_misses += 1
            else:
                self.top_misses += 1
            return None

    def track_player(self, detections, last_det, min_score=0.85, bottom=True):

        num_detections = len(detections[0]['boxes'])
        if num_detections > 0:
            scores = detections[0]['scores'][:].cpu().detach().numpy()
            index = np.where(scores > min_score)[0]
            candidate_scores = np.zeros((len(index), 1))

            if len(index) > 0:
                for i in range(len(index)):
                    bbox = detections[0]['boxes'][index[i]].cpu().detach().numpy()
                    score = detections[0]['scores'][index[i]].cpu().detach().numpy()
                    distance_to_last_candidate = distance_to_prev_detection(last_bbox=last_det, candidate_bbox=bbox)
                    candidate_scores[i, :] = (1 / distance_to_last_candidate) * 0.5 + score * 0.5

                best_candidate = np.argmax(candidate_scores)
                bbox = detections[0]['boxes'][best_candidate].cpu().detach().numpy()
                keypoints = detections[0]['keypoints'][best_candidate].cpu().detach().numpy()
                return [keypoints, bbox]
            else:
                print('Bottom persons tracking not performed! Candidate with small confidence!')
                if bottom:
                    self.bottom_misses += 1
                else:
                    self.top_misses += 1
                return None
        else:
            print('Bottom persons tracking not performed! NO candidate detected!')
            if bottom:
                self.bottom_misses += 1
            else:
                self.top_misses += 1
            return None

    def get_iteration_mask(self, mask, last_box, margin):
        x0, y0, x1, y1 = last_box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        # Define new searching window to find detection.
        min_b = max(x0 - margin, 0)
        max_b = min(x1 + margin, self.width)

        min_t = max(y0 - margin, 0)
        max_t = min(y1 + margin, self.height)

        iteration_mask = np.zeros_like(mask)
        iteration_mask[min_t:max_t, min_b:max_b, :] = mask[min_t:max_t, min_b:max_b, :]

        return iteration_mask

    def find_player_top(self):

        # Initialize variables
        # self.player_top_boxes = [None] * (self.total_fps - 1)
        keypoints = [None] * self.total_fps

        # Get the first no None element in players ID's list to start tracking.
        playerID = next((element for element in self.top_player_IDs if element is not None), None)
        fisrt_player = next((index for index, element in enumerate(self.top_player_IDs) if element is not None), None)

        first_appearance_ids = list(set(self.candidate_first_appearance.keys()))
        first_appearance_frame = list(set(self.candidate_first_appearance.values()))

        for frame_idx in range(fisrt_player, self.total_fps - 1):
            print('Framce id: ', frame_idx)
            top_player_bbox = self.top_candidate_boxes[playerID][frame_idx]
            # Get last detected bounding box
            cumulated_bboxes = self.top_candidate_boxes[playerID][fisrt_player:frame_idx]
            last_bbox = next((element for element in reversed(cumulated_bboxes) if element is not None), None)
            print('Last Bounding box: ', last_bbox)

            if top_player_bbox is not None:
                cumulated_hists = self.top_candidate_appearance[playerID][fisrt_player:frame_idx]
                last_hist = next((element for element in reversed(cumulated_hists) if element is not None), None)

                if frame_idx in first_appearance_frame and last_bbox is not None and frame_idx > 0:
                    playerID = self.reId(last_bbox=last_bbox, last_Hist=last_hist,
                                         frame_idx=frame_idx, playerID=playerID)
            else:
                cumulated_hists = self.top_candidate_appearance[playerID][fisrt_player:frame_idx]
                last_hist = next((element for element in reversed(cumulated_hists) if element is not None), None)

                playerID = self.reId(last_bbox=last_bbox, last_Hist=last_hist, frame_idx=frame_idx, playerID=playerID)

            self.player_top_boxes[frame_idx] = self.top_candidate_boxes[playerID][frame_idx]
            keypoints[frame_idx] = self.top_candidate_keypoints[playerID][frame_idx]

        self.player_top_keypoints = keypoints

    def reId(self, last_bbox, last_Hist, frame_idx, playerID):

        # Initialize parameters for re-Identification.
        first_appearance_ids = list(set(self.candidate_first_appearance.keys()))

        last_playerBbox = last_bbox
        last_playerHist = last_Hist

        fist_playerHist = self.top_player_first_h
        max_score = 0

        for candidate_id in first_appearance_ids:
            candidate_hist = self.top_candidate_appearance[candidate_id][frame_idx]
            if candidate_hist is not None:

                norm_hist2 = fist_playerHist / np.sum(fist_playerHist)

                # Normalize histograms
                norm_hist1 = last_playerHist / np.sum(last_playerHist)
                # Combine and normalize histograms using weighted averaging
                combined_hist = (0.65 * norm_hist1) + (0.35 * norm_hist2)
                combined_hist /= np.sum(combined_hist)

                candidateBbox = self.top_candidate_boxes[candidate_id][frame_idx]
                prev_distance = distance_to_prev_detection(last_bbox=last_playerBbox, candidate_bbox=candidateBbox)
                distance = min(1 / prev_distance, 1)
                bc_score = min(calculate_bhattacharyya_distance(hist1=norm_hist1, hist2=norm_hist2), 1)

                # Combine Inverse Bhattacharyya Distance and distance to previous player bbox distance to get score.
                score = bc_score * 0.6 + distance * 0.4
                print('Candidate id: ', candidate_id, ' Score: ', score, ' with respect to: ',
                      playerID)

                candidate_area = area_of_box(box=candidateBbox)
                lastDect_area = area_of_box(box=last_bbox)

                area_diff = min(candidate_area, lastDect_area) / max(candidate_area, lastDect_area)
                print('Area difference: ', area_diff)
                print('Distance: ', prev_distance)
                if score >= max_score and prev_distance < 50. and area_diff > 0.3:
                    max_score = score
                    playerID = candidate_id

        return playerID

    def mark_player_box(self, frame, boxes, frame_num, bottom=False):
        box = boxes[frame_num]
        if box is not None:
            if bottom:
                keypoints = self.player_bottom_keypoints[frame_num]
            else:
                keypoints = self.player_top_keypoints[frame_num]

            frame = utils_pose.draw_keypoints_and_boxes(keypoints=keypoints, bbox=box, image=frame)
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 0], 2)
        return frame

    def detect(self, orig_frame, image_court):

        pil_image = Image.fromarray(image_court).convert('RGB')

        # transform image
        image = self.transform(pil_image)
        # add a batch dimension
        image = image.unsqueeze(0).to(self.device)

        # get detections
        with torch.no_grad():
            outputs = self.model(image)

        return outputs
