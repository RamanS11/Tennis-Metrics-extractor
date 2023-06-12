import numpy as np
import cv2
import os

from config import cfg
code_config = cfg()


########################################################################################################################
#                                               DEFINE USEFUL FUNCTIONS                                                #
########################################################################################################################
def calculator_error_matriz(H_pre, H_gt):
    # Compute matrix error
    H_error = np.linalg.inv(H_gt) @ H_pre

    # Compute norm of matrix error
    error_matriz = np.linalg.norm(H_error, 'fro')
    print('Error matriz: ', error_matriz)

    return error_matriz


def normalize_gt_matrix(gt_matrix):
    third_element = gt_matrix[-1, -1]
    return gt_matrix / third_element


def recompute_H_pred(H_predicted, reference_points: list):
    """
    Function that re-computes H_predicted using the four outside points as reference (to compare it with GT)
    :param H_predicted: Predicted homography
    :param reference_points: List with coordinates of outside points in reference court
    :return: H_predicted_out
    """
    ref_points_array = np.asarray(reference_points)
    proj_points = np.empty((4, 2))

    for i, point in enumerate(ref_points_array):
        projected_point_homogeneus = np.dot(H_predicted, point)
        projected_point_2d = (projected_point_homogeneus[0] / projected_point_homogeneus[-1],
                              projected_point_homogeneus[1] / projected_point_homogeneus[-1])
        proj_points[i, :] = np.asarray(projected_point_2d)

    new_predicted_H, _ = cv2.findHomography(np.float32(proj_points), np.float32(ref_points_array),
                                            cv2.RANSAC, 5.0)
    return new_predicted_H


def get_plan_view(H, img):
    return cv2.warpPerspective(img, H, (gt_img.shape[1], gt_img.shape[0]))
########################################################################################################################
#                                         END DEFINING USEFUL FUNCTIONS                                                #
########################################################################################################################


########################################################################################################################
#                                     LOAD GROUND TRUTH AND PREDICTED VARIABLES                                        #
########################################################################################################################
filled_court_path = os.path.join(code_config.get_COURT_CONF_DIR(), 'court_reference_fill.png')
filled_court = cv2.imread(filled_court_path, cv2.IMREAD_GRAYSCALE)

# Define name of video to compute metrics from.
result_folders = os.listdir(code_config.get_COURT_RESULTS_DIR())
for res in result_folders:
    tmp_res_dir = os.path.join(code_config.get_COURT_GT_DIR(), res)
    if not os.path.isdir(tmp_res_dir):
        os.mkdir(tmp_res_dir)
name = result_folders[4]

# Load ground truth projection and homography
source = os.path.join(code_config.get_COURT_GT_DIR(), name)
gt_img_path = os.path.join(source, 'gt_plan_view.png')
gt_img = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)

gt_h_path = os.path.join(source, 'H_gt.npy')
gt_h = np.load(gt_h_path)

gt_h_inv_path = os.path.join(source, 'H_inv_gt.npy')
gt_h_inv = np.load(gt_h_inv_path)

# Load predicted projection and homography
destination = os.path.join(code_config.get_COURT_RESULTS_DIR(), name)
predicted_h_path = os.path.join(destination, 'homography_matrix.npy')
predicted_h = np.load(predicted_h_path)
predicted_img = get_plan_view(predicted_h, filled_court)

# Show predicted and ground truth projections
cv2.imshow('GT projection', gt_img)
cv2.waitKey(0)

cv2.imshow('predicted projection', predicted_img)
cv2.waitKey(0)
########################################################################################################################
#                                     LOAD GROUND TRUTH AND PREDICTED VARIABLES                                        #
########################################################################################################################


########################################################################################################################
#                               NORMALIZE GROUND TRUTH AND RECOMPUTE PREDICTED HOMOGRAPHY                              #
########################################################################################################################
# Define four outside points in reference court.
p1 = (285.5, 560.5, 1)
p2 = (1380.5, 560.5, 1)
p3 = (285.5, 2936.5, 1)
p4 = (1380.5, 2936.5, 1)
ref_points = [p1, p2, p4, p3]

# Re-compute predicted homography matrix using four outside points as reference (to compare it with ground truth)
H_predicted_2 = recompute_H_pred(H_predicted=predicted_h, reference_points=ref_points)

# Normalize ground truth matrix to better compare with predicted homography.
normalized_gt_h = normalize_gt_matrix(gt_matrix=gt_h)
########################################################################################################################
#                             END NORMALIZE GROUND TRUTH AND RECOMPUTE PREDICTED HOMOGRAPHY                            #
########################################################################################################################


########################################################################################################################
#                                           COMPUTE DIFFERENT METRICS                                                  #
########################################################################################################################
# Compute Intersection over Union (IoU) between ground truth and predicted projections.
intersection = cv2.bitwise_and(gt_img, predicted_img)
union = cv2.bitwise_or(gt_img, predicted_img)
IoU = intersection.sum() / union.sum()
print('Intersection over Union in ', name, ' : ', IoU*100, ' %')

# Compute matrix error (distance between matrices)
calculator_error_matriz(H_predicted_2, normalized_gt_h)

# Print both recomputed predicted homography and normalized ground truth matrix.
print('Predicted homography matrix: ', H_predicted_2)
print('Ground truth inverse homography matrix: ', normalized_gt_h)
########################################################################################################################
#                                        END COMPUTE DIFFERENT METRICS                                                 #
########################################################################################################################
