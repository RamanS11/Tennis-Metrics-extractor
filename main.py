import numpy as np
import pickle
import torch
import json
import cv2
import os

from source.court_detection.CourtDetector import courtDetection, get_line_center
from source.court_detection.CourtRefernce import CourtReference

from source.pose_estimation.poseEstimation import playersDetection, dump_data
from source.pose_estimation.utils import calculate_feet_positions

from source.ball_tracking.TrackNet_BallTracker import BallTracking
from source.ball_tracking.custom_BallTracker import BallTracking_improved

from utils import get_video_properties, release_video, guardar_video, resize_upper_view
from config import cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU Use : ', torch.cuda.is_available())

code_config = cfg()
globals()

# Read input video
videos = ['game1_Clip4.mp4', 'game2_Clip8.mp4', 'game3_Clip7.mp4', 'video_input2.mp4', 'video_input3.mp4',
          'video_input4.mp4', 'video_input5.mp4', 'video_input6.mp4', 'video_input8.mp4', 'point_1.mp4', 'point_5.mp4']

video_name = videos[-2]

code_config.set_result_directory(video_name)
"""
# Define ball tracking variable and execute tracking
ballTracker = BallTracking(video_name=video_name, Cfg=code_config)
ballTracker.track_ball()

c_ballTracker = BallTracking_improved(video_name=video_name, Cfg=code_config)
c_ballTracker.custom_tracking()
"""

# Get average mask from the three first frames of the video
lines, bottom_mask, top_mask, H, invH = courtDetection(video_name)

playerDetector = playersDetection(video_name=video_name, Cfg=code_config)
referenceCourt = CourtReference()

net_line = lines[2 * 4:3 * 4]
baseline_top = lines[:4]
baseline_bottom = lines[4:8]

net_center = get_line_center(net_line)
baseline_top_center = get_line_center(baseline_top)
baseline_bottom_center = get_line_center(baseline_bottom)

# Set video output path and start video capture
code_config.set_destination_directory(video_name)
video = os.path.join(code_config.get_INPUT_VIDEO_DIR(), video_name)
vid = cv2.VideoCapture(video)
frame_count = 0

# get videos properties
fps, length, v_width, v_height = get_video_properties(vid)

# Define top view video Writer.
court = referenceCourt.court.copy()
court = cv2.line(court, *referenceCourt.net, 255, 5)
c_width, c_height = court.shape[::-1]

save_path = os.path.join(code_config.get_RESULT_VIDEO_OUTPUT(), 'top_view.mp4')
top_view = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (c_width // 2, c_height // 2))

# create a dictionary
bottom_player_coord = {'detections': '', 'frame_idx': -1}
top_player_coord = {'detections': '', 'frame_idx': -1}

return_value, frame = vid.read()
number_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

return_value, frame = vid.read()
frame_idx = frame_count
# Print court lines over frame
for i in range(0, len(lines), 4):
    x1, y1, x2, y2 = lines[i], lines[i + 1], lines[i + 2], lines[i + 3]
    cv2.line(frame, (x1, y1), (x2, y2), (220, 220, 220), 5)

cv2.imshow('court_line', frame)
cv2.waitKey(0)

frames = []
original_frames = []

top_Detections = []

while return_value:

    return_value, frame = vid.read()

    if return_value:

        original_frame = frame.copy()
        original_frames.append(original_frame)
        bottom_court = original_frame.copy()
        bottom_court[bottom_mask == 0, :] = (0, 0, 0)
        playerDetector.detect_player_bottom(frame=frame, bottom_mask=bottom_court, baseline=baseline_bottom,
                                            frame_idx=frame_count)

        top_court = original_frame.copy()
        top_court[top_mask == 0, :] = (0, 0, 0)
        top_boxes = playerDetector.detect_candidates_top(frame=frame, top_mask=top_court, baseline=baseline_top,
                                                         frame_idx=frame_count)
        # Print court lines over frame
        for i in range(0, len(lines), 4):
            x1, y1, x2, y2 = lines[i], lines[i + 1], lines[i + 2], lines[i + 3]
            cv2.line(frame, (x1, y1), (x2, y2), (100, 100, 100), 2)

        new_frame = cv2.resize(frame, (v_width, v_height))
        frames.append(new_frame)
        cv2.imshow('top_boxes', top_boxes)
        cv2.waitKey(1)
        top_Detections.append(top_boxes)
        print('Framce id: ', frame_count)
        frame_count += 1
    else:
        print('Video has ended or failed')
        break

# Release video object.
release_video(vid)
guardar_video(frames=top_Detections, nombre_video=os.path.join(code_config.get_RESULT_VIDEO_OUTPUT(), 'detection_top'),
              fps=fps)
'''
with open('candidate_bboxes.pkl', 'rb') as fp:
    top_bboxes = pickle.load(fp)

with open('candidate_hists.pkl', 'rb') as fp:
    top_hists = pickle.load(fp)

playerDetector.top_candidate_boxes = top_bboxes
playerDetector.top_candidate_appearance = top_hists
playerDetector.best_top_candidate_id = 1
'''
# empty Torch cache memory
torch.cuda.empty_cache()

# Select player in order to codify texture information from candidates detected.
playerDetector.find_player_top()

boxes_p1 = playerDetector.player_bottom_boxes
boxes_p2 = playerDetector.player_top_boxes


# Guardar el diccionario con bboxes del jugador 1 en un archivo JSON
path_result_p1 = os.path.join(code_config.get_RESULT_VIDEO_OUTPUT(), 'coords_p1.json')
bounding_boxes_serializable_1 = {key: value.tolist() for key, value in boxes_p1.items()}
with open(path_result_p1, 'w') as file:
    json.dump(bounding_boxes_serializable_1, file)

# Guardar el diccionario con bboxes del jugador 2 en un archivo JSON
path_result_p2 = os.path.join(code_config.get_RESULT_VIDEO_OUTPUT(), 'coord_p2.json')
bounding_boxes_serializable_2 = {key: value.tolist() if isinstance(value, np.ndarray) else value
                                 for key, value in boxes_p2.items() if value is not None}
with open(path_result_p2, 'w') as file:
    json.dump(bounding_boxes_serializable_2, file)

# initialize frame counters
frame_number = 0
orig_frame = 0
out = playerDetector.get_video_writer()

print('bboxes p1: ', len(boxes_p1))
print('bboxes_p2: ', len(boxes_p2))

top_coords_p1, top_coords_p2 = calculate_feet_positions(invH=invH, player_1_boxes=boxes_p1,
                                                        keypoints_p1=playerDetector.player_bottom_keypoints,
                                                        player_2_boxes=boxes_p2)

original_court = referenceCourt.court.copy()

for img in frames:

    img = playerDetector.mark_player_box(img, boxes_p1, frame_number, bottom=True)
    img = playerDetector.mark_player_box(img, boxes_p2, frame_number, bottom=False)
    img = cv2.resize(img, (v_width, v_height))

    feet_pos_p1 = (int(top_coords_p1[frame_number, 0]), int(top_coords_p1[frame_number, 1]))
    feet_pos_p2 = (int(top_coords_p2[frame_number, 0]), int(top_coords_p2[frame_number, 1]))

    _court = referenceCourt.court.copy()
    upper_view = cv2.circle(_court, feet_pos_p1, 45, (255, 255, 0), -1)
    if feet_pos_p2[0] is not None:
        upper_view = cv2.circle(upper_view, feet_pos_p2, 45, (255, 255, 255), -1)

    upper_view = resize_upper_view(input_image=upper_view)

    top_view.write(upper_view.astype(np.uint8))
    cv2.imshow('track_top', img)
    cv2.waitKey(40)
    out.write(img)
    frame_number += 1

# top_view.release()
out.release()
playerDetector.release_video()
