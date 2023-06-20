import cv2
import numpy as np
import matplotlib.colors
from scipy import signal

# NOTE -> CAMBIADO INTERNAMENTE LIBRERIA DE MATPLOTLIB PARA QUE NO PETEN DEPENDENCIAS CON NUMPY!
# volver a cambiar en [conda env -> matplotlib -> init file -> check versions function.

# pairs of edges for 17 of the keypoints detected ...
# ... these show which points to be connected to which point ...
# ... we can omit any of the connecting points if we want, basically ...
# ... we can easily connect less than or equal to 17 pairs of points ...
# ... for keypoint RCNN, not  mandatory to join all 17 keypoint pairs
# Note -> Left ankle: 16, Right ankle: 17

edges = [(0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
         (12, 14), (14, 16), (5, 6)]


def draw_keypoints_and_boxes(keypoints, bbox, image):

    try:
        keypoints = keypoints[:, :].reshape(-1, 3)
        for p in range(keypoints.shape[0]):
            # draw the keypoints
            cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])),
                       3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        # draw the lines joining the keypoints
        for ie, e in enumerate(edges):
            # get different colors for the edges
            rgb = matplotlib.colors.hsv_to_rgb([ie/float(len(edges)), 1.0, 1.0])
            # hsv = np.array([[ie/float(len(edges)) * 179, 1.0 * 255, 1.0 * 255]], dtype=np.uint8)
            # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            rgb = rgb*255
            # join the keypoint pairs to draw the skeletal structure
            cv2.line(image, (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                     (int(keypoints[e, 0][1]), int(keypoints[e, 1][1])),
                     tuple(rgb), 2, lineType=cv2.LINE_AA)
    except TypeError:
        print('No keypoints detected!')
        pass

    try:
        # draw the bounding boxes around the objects
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 255, 0), thickness=2)
    except TypeError:
        print('Bounding Box: ', bbox)
        pass

    return image


def calculate_feet_positions(invH, player_1_boxes, keypoints_p1, player_2_boxes):
    """
    Calculate the feet position of both players using the inverse transformation of the court and the boxes
    of both players
    """
    num_frames = len(player_1_boxes)

    positions_1 = [None] * num_frames
    detected_p1 = [False] * num_frames

    # Bottom player feet locations
    for i, box in player_1_boxes.items():
        if box is not None:
            left_knee = keypoints_p1[i][15]
            right_knee = keypoints_p1[i][16]

            if left_knee[-1] and right_knee[-1]:
                # Compute mean knees positions (if visible) and transform using inverse Histogram matrix.
                avg_x = int(left_knee[0] + (right_knee[0] - left_knee[0]) / 2)
                avg_y = int(left_knee[1] + (right_knee[1] - left_knee[1]) / 2)
                feet_pos = np.array([avg_x, avg_y], dtype=np.float32).reshape((1, 1, 2))

            else:
                # Compute player position using bottom bbox line (if knees are not visible!)
                feet_pos = np.array([(box[0] + (box[2] - box[0]) / 2).item(), box[3].item()],
                                    dtype=np.float32).reshape((1, 1, 2))

            feet_court_pos = cv2.perspectiveTransform(feet_pos, invH).reshape(-1)
            positions_1[i] = feet_court_pos
            detected_p1[i] = True

    # Smooth both feet locations
    positions_1 = np.array(positions_1)
    smoothed_1 = np.zeros_like(positions_1)
    smoothed_1[:, 0] = signal.savgol_filter(positions_1[:, 0], 7, 2)
    smoothed_1[:, 1] = signal.savgol_filter(positions_1[:, 1], 7, 2)
    smoothed_1[not detected_p1, :] = [None, None]

    positions_2 = [None] * num_frames
    detected_p2 = [False] * num_frames

    # Top player feet locations
    for i, box_2 in player_2_boxes.items():

        if box_2 is not None:

            # Compute player position using bottom bbox line (if knees are not visible!)
            feet_pos = np.array([(box_2[0] + (box_2[2] - box_2[0]) / 2).item(), box_2[3].item()],
                                dtype=np.float32).reshape((1, 1, 2))

            feet_court_pos = cv2.perspectiveTransform(feet_pos, invH).reshape(-1)
            positions_2[i] = feet_court_pos
            detected_p2[i] = True
        elif i > 1:
            try:
                positions_2[i] = positions_2[i-1]
            except IndexError:
                print('Index: ', i, ' error!, total number of detections: ', len(player_2_boxes.keys()))
        else:
            positions_2[i] = np.array([0, 0])

    positions_2 = np.array(positions_2)
    smoothed_2 = np.zeros_like(positions_2)
    smoothed_2[:, 0] = signal.savgol_filter(positions_2[:, 0], 7, 2)
    smoothed_2[:, 1] = signal.savgol_filter(positions_2[:, 1], 7, 2)
    smoothed_2[not detected_p2, :] = [None, None]

    return smoothed_1, smoothed_2

