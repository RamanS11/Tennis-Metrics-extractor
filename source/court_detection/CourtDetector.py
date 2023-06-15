import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from source.court_detection.CourtLineCandidateDetector import CourtLineCandidateDetector
from source.court_detection.TennisCourtFitter import TennisCourtFitter, _threshold
from source.court_detection.CourtRefernce import CourtReference
from config import cfg

code_config = cfg()


def get_line_center(line):
    """
    Given input line this function computes center point
    :param line: vector with top left and lower right bounding box coordinates.
    :return: input line's center point.
    """
    x1, y1, x2, y2 = line
    line_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

    return line_center


def crop_avg_mask(mask):
    """
    Given a list with court mask extracted from the first three frames, extract average mask.
    :param mask: List containing threshold image of the first three frames of the video.
    :return: binary mask with enough information to detect lines and fit court model.
    """
    # Convert list to numpy array and extract frame dimensions.
    masks = np.asarray(mask)
    _, w, h = masks.shape
    msk = np.zeros((w, h))

    # Generate average mask from threshold images.
    for m in mask:
        msk += m

    msk_max = msk.max().max()
    msk_norm = msk / msk_max
    msk_norm[msk_norm < 0.5] = 0
    msk_norm[msk_norm >= 0.5] = 1

    # Crop average mask to discard future outliers in line detection.
    avg_msk = np.zeros((w, h))
    avg_msk[w // 8:7 * w // 8, h // 8:7 * h // 8] = msk_norm[w // 8:7 * w // 8, h // 8:7 * h // 8]

    return avg_msk


def courtDetection(video_name):

    # Define courtLineDetector, tennisCourtFitter and referenceCourt class variables
    courtLineDetector = CourtLineCandidateDetector(debug=False)
    tennisCourtFitter = TennisCourtFitter(debug=False)
    referenceCourt = CourtReference()

    # Read input video
    video = os.path.join(code_config.get_INPUT_VIDEO_DIR(), video_name)
    code_config.set_destination_directory(video_name)
    vid = cv2.VideoCapture(video)
    return_value, frame = vid.read()

    # Define variables to obtain average mask from first three frames in of input video.
    mask = []
    cont = 0

    while True and cont <= 2:
        ret, frame = vid.read()

        if ret:
            cont += 1
            binary_img = _threshold(frame)
            binary_img = cv2.GaussianBlur(binary_img, (3, 3), 0)
            mask.append(binary_img)

    avg_msk = crop_avg_mask(mask=mask)

    # Get horizontal and vertical lines from average mask.
    horizontal, vertical = courtLineDetector.CourtLineCandidateDetector(avg_msk, frame)
    if horizontal is not None:
        print('Vertical lines: ', len(vertical), ' horizontal lines: ', len(horizontal))
    else:
        # If no lines detected return None and break whole program.
        print('No lines detected!')
        return None, None, None

    # Find the best homography matrix that fits reference court with detected lines.
    lines, H, invH = tennisCourtFitter.TennisCourtFitter(candidate_lines=[horizontal, vertical], binaryImage=avg_msk,
                                                         frame=frame, final_dir=code_config.get_FINAL_DIR())

    # Save homography and inverse homography matrices into results directory.
    name = 'homography_matrix.npy'
    if not os.path.exists(code_config.get_FINAL_DIR()):
        os.mkdir(code_config.get_FINAL_DIR())
    np.save(os.path.join(code_config.get_FINAL_DIR(), name), H)

    name = 'homography_inv_matrix.npy'
    if not os.path.exists(code_config.get_FINAL_DIR()):
        os.mkdir(code_config.get_FINAL_DIR())
    np.save(os.path.join(code_config.get_FINAL_DIR(), name), H)

    # Project reference court into input frame dimensions and vice versa (and save results).
    court_projection = cv2.warpPerspective(referenceCourt.court, H, (frame.shape[1], frame.shape[0]))
    plan_view = cv2.warpPerspective(frame, invH, (referenceCourt.court.shape[1], referenceCourt.court.shape[0]))

    result_dir = os.path.join(code_config.get_COURT_RESULTS_DIR(), video_name.split('.')[0])
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    cv2.imwrite(os.path.join(code_config.get_FINAL_DIR(), 'court_projection.png'), court_projection)
    cv2.imwrite(os.path.join(code_config.get_FINAL_DIR(), 'planar_view.png'), plan_view)

    # Extract bottom court and upper court masks to further player keypoint detections and tracking.
    white_ref_bottom = referenceCourt.get_court_mask(mask_type=1)
    white_mask_bottom = cv2.warpPerspective(white_ref_bottom, H, frame.shape[1::-1])

    white_ref_top = referenceCourt.get_court_mask(mask_type=2)
    white_mask_top = cv2.warpPerspective(white_ref_top, H, frame.shape[1::-1])
    white_mask_top = cv2.dilate(white_mask_top, np.ones((75, 1)), anchor=(0, 0))

    # Release video and close all windows.
    vid.release()

    return lines, white_mask_bottom, white_mask_top, H, invH
