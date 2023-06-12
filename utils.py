import cv2
import numpy as np
# import matplotlib

# pairs of edges for 17 of the keypoints detected ...
# ... these show which points to be connected to which point ...
# ... we can omit any of the connecting points if we want, basically ...
# ... we can easily connect less than or equal to 17 pairs of points ...
# ... for keypoint RCNN, not  mandatory to join all 17 keypoint pairs


def release_video(vid_Object):
    # release video capturer
    vid_Object.release()
    # close all frames and video windows
    cv2.destroyAllWindows()


def get_video_properties(video):
    # Find OpenCV version
    fps = video.get(cv2.CAP_PROP_FPS)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    v_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, length, v_width, v_height
