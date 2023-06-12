import numpy as np
import cv2 as cv
import os

from config import cfg
code_config = cfg()

drawing = False             # True if mouse is pressed
src_x, src_y = -1, -1
dst_x, dst_y = -1, -1

src_list = []
dst_list = []


# mouse callback function
def select_points_src(event, x, y, flags, pts):
    global src_x, src_y, drawing
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        src_x, src_y = x, y
        cv.circle(src_copy, (x, y), 5, (0, 0, 255), -1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False


# mouse callback function
def select_points_dst(event, x, y, flags, pts):
    global dst_x, dst_y, drawing
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        dst_x, dst_y = x, y
        cv.circle(dst_copy, (x,y), 5, (0, 0, 255), -1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False


def get_plan_view(src, dst):
    src_pts = np.array(src_list).reshape(-1, 1, 2)
    dst_pts = np.array(dst_list).reshape(-1, 1, 2)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    print("H:")
    print(H)

    inv_H = cv.invert(H)[1]
    final_dir_h = os.path.join(final_directory, 'H_inv_gt')
    final_dir_h_inv = os.path.join(final_directory, 'H_gt')

    np.save(final_dir_h, H)
    np.save(final_dir_h_inv, inv_H)

    return cv.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))


def merge_views(src, dst):
    plan_view = get_plan_view(src, dst)
    for i in range(0, dst.shape[0]):
        for j in range(0, dst.shape[1]):
            if plan_view.item(i, j, 0) == 0 and plan_view.item(i, j, 1) == 0 and plan_view.item(i, j, 2) == 0:
                plan_view.itemset((i, j, 0), dst.item(i, j, 0))
                plan_view.itemset((i, j, 1), dst.item(i, j, 1))
                plan_view.itemset((i, j, 2), dst.item(i, j, 2))
    return plan_view


result_folders = os.listdir(code_config.get_COURT_RESULTS_DIR())

for res in result_folders:
    tmp_res_dir = os.path.join(code_config.get_COURT_GT_DIR(), res)
    if not os.path.isdir(tmp_res_dir):
        os.mkdir(tmp_res_dir)

source = os.path.join(code_config.get_COURT_CONF_DIR(), 'court_reference_fill.png')
src = cv.imread(source, -1)
src_copy = src.copy()
cv.namedWindow('src')
cv.moveWindow("src", 80, 80)
cv.setMouseCallback('src', select_points_src)

name = result_folders[4]
destination = os.path.join(code_config.get_COURT_RESULTS_DIR(), name)

predicted_h_path = os.path.join(destination, 'homography_matrix.npy')
predicted_h = np.load(predicted_h_path)

destination_img = os.path.join(destination, 'frame.png')
dst = cv.imread(destination_img, -1)
dst_copy = dst.copy()
cv.namedWindow('dst')
cv.moveWindow("dst", 780, 80)
cv.setMouseCallback('dst', select_points_dst)

final_directory = os.path.join(code_config.get_COURT_GT_DIR(), name)

while 1:
    cv.imshow('src', src_copy)
    cv.resize(src_copy, (src_copy.shape[0]//2, src_copy.shape[1]//2))
    cv.imshow('dst', dst_copy)
    k = cv.waitKey(1) & 0xFF
    if k == ord('s'):
        print('save points')
        cv.circle(src_copy, (src_x, src_y), 5, (0, 255, 0), -1)
        cv.circle(dst_copy, (dst_x, dst_y), 5, (0, 255, 0), -1)
        src_list.append([src_x, src_y])
        dst_list.append([dst_x, dst_y])
        print("src points:")
        print(src_list)
        print("dst points:")
        print(dst_list)
    elif k == ord('h'):
        print('create plan view')
        plan_view = get_plan_view(src, dst)
        cv.imshow("plan view", plan_view)
        cv.imwrite(os.path.join(final_directory, 'gt_plan_view.png'), plan_view)
    elif k == ord('m'):
        print('merge views')
        merge = merge_views(src, dst)
        cv.imshow("merge", merge)
    elif k == 27:
        break

cv.destroyAllWindows()
