# Module with functions regarding:
#     1. Detection of outliers in TrackNetV2 predictions.
#     2. Inference of missing data when no data predicted.
#     3. Obtain metrics comparing vanilla TrackNetV2 and combination of model + missing data inference over videos with
#         annotations (quantitative results).
#         For videos without any annotation given -> qualitative results (no annotations to compare performance).
# Author: Ramanjit Singh Kaur (module used for student's Final Degree Project)

# libraries imports
import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import argrelextrema, argrelmin, argrelmax

# own imports
from source.ball_tracking.KalmanFilter import KalmanFilter
from utils import get_video_properties


class label_instance:
    name = ""
    vis = 0
    x = 0
    y = 0
    status = 0

    def __init__(self, name, vis, x, y, status):
        try:
            self.name = name
            self.vis = int(vis)
            self.x = int(x)
            self.y = int(y)
            self.status = str(status)
        except ValueError:
            pass


def get_metrics(gt, prediction):
    """
    Function that will return list with all frames with missing data.
    We consider missing data if we have the annotation in the ground truth, but no prediction is given from TrackNetV2.
    :param gt: List with labels corresponding to Ground Truth annotations in specific video.
    :param prediction: List with labels corresponding to predictions annotation done over specific video.
    :return: list with file names where we have missing data.
    """
    files_md = []
    se = 0
    for i in range(len(prediction)):

        # Check if both gt and prediction annotations are available.
        a_vis = gt[i].vis
        p_vis = prediction[i].vis
        vis = a_vis == p_vis == 1

        # Check if both gt and predicted instances corresponds to same frame.
        p_frame = int(prediction[i].name)
        g_frame = int(gt[i].name.split('.jpg')[0])
        same_frame = p_frame == g_frame
        try:
            assert same_frame, "Not working with same gt and prediction frames!"
            gt_i = np.array([gt[i].x, gt[i].y])
            p_i = np.array([prediction[i].x, prediction[i].y])
            res = np.sqrt((p_i[0] - gt_i[0]) ** 2 + (p_i[1] - gt_i[1]) ** 2)
            se += res

        except AssertionError:
            files_md.append(gt[i].name)
            pass

    mse = se / len(gt)
    vis_pred = [p for p in prediction if p.vis != 0 and (p.x != 0 and p.y != 0)]
    print('Total number of annotations (gt): ', len(gt))
    print('Total number of predictions: ', len(vis_pred))
    print('Final mean square error: ', mse)
    print('\n')


def max_intra_dist(ann_list, gt=True):
    """
    Function that returns maximum displacement (in x-axis and y-axis)
    from list of annotations given as input (only between visible annotations)
    :param gt: boolean indicating weather input data is from Ground truth or not.
    :param ann_list: list with annotations (gt) obtained from '.csv' file.
    :return: maximum distance (in both axis) intra-frame
    """
    ann_valid_list = []
    [ann_valid_list.append(ann) for ann in ann_list if ann.vis != 0]
    max_x = abs(ann_valid_list[1].x - ann_valid_list[0].x)
    max_y = abs(ann_valid_list[1].y - ann_valid_list[0].y)

    for i in range(1, len(ann_valid_list)):
        if not gt:
            # If not ground truth, check if frames are consecutive!
            crt_idx = int(ann_valid_list[i].name)
            prev_idx = int(ann_valid_list[i - 1].name)
            idx_dif = crt_idx - prev_idx
            try:
                assert idx_dif == 1, "no consecutive frames"
            except AssertionError:
                pass
        if ann_valid_list[i].vis:
            max_x = max(abs(ann_valid_list[i].x - ann_valid_list[i - 1].x), max_x)
            max_y = max(abs(ann_valid_list[i].y - ann_valid_list[i - 1].y), max_y)

    return max_x, max_y


def outliers_detector(prediction, max_d_x, max_d_y):
    """
    Function that will detect outliers in predictions list.
    Using a window (defined by max_d_x and max_d_y)
    :param max_d_y: maximum distance allowed to have a displacement between consecutive frames in x-axis.
    :param max_d_x: maximum distance allowed to have a displacement between consecutive frames in y-axis.
    :param prediction: list with predictions.
    :return: list with predictions (without outliers), and list with outliers detected.
    """
    # Only evaluate over available annotations
    valid_prediction = []
    visibility = [False] * len(prediction)

    for p in prediction:
        if p.vis == 1:
            valid_prediction.append(p)
            visibility[int(p.name)] = True

    max_d_x += 5
    max_d_y += 5
    no_consecutive = 0
    outliers = []
    prediction_ = prediction

    for v in range(1, len(valid_prediction)):
        crt_idx = int(valid_prediction[v].name)
        prev_idx = int(valid_prediction[v - 1].name)
        idx_dif = crt_idx - prev_idx

        try:
            assert idx_dif == 1, "no consecutive frames"
            assert valid_prediction[v - 1].status != 'outlier'

            local_dist_x = abs(valid_prediction[v].x - valid_prediction[v - 1].x)
            local_dist_y = abs(valid_prediction[v].y - valid_prediction[v - 1].y)

            if local_dist_x > max_d_x or local_dist_y > max_d_y:
                # Re-define outlier in prediction list.
                outlier_idx = prediction_.index(valid_prediction[v])

                prediction_[outlier_idx].vis = 0
                prediction_[outlier_idx].x = 0
                prediction_[outlier_idx].y = 0
                prediction_[outlier_idx].status = 'outlier'
                outliers.append(valid_prediction[v])

        except AssertionError:
            no_consecutive += 1
            pass

    # print('number of outliers detected: {}'.format(len(outliers)))
    print('final max_x and max_y: {} {}'.format(max_d_x, max_d_y))

    return prediction_, outliers


def get_y_cords(new_predictions, show=False):
    """
    Function that returns y-axis coordinates from list of annotations
    :param show: boolean indicating whether show or NOT results in real time.
    :param new_predictions: list of input predictions
    :return: list with y-coordinates extracted from new_predictions
    """
    y_cords = []
    [y_cords.append(npp.y) for npp in new_predictions if npp.vis == 1]

    z = np.arange(len(new_predictions))

    if show:
        plt.plot(z, np.array(y_cords))
        plt.show()

    return np.array(y_cords)


def get_local(coord_y: np.ndarray):
    """
    Returns the max and mins of and 1-D array
    :param coord_y: 1-D numpy array
    :return: local maxima's, minimal
    """
    maxima = argrelextrema(np.array(coord_y), np.greater, order=2)
    minima = argrelextrema(np.array(coord_y), np.less, order=2)

    return maxima[0], minima[0]


def get_local_2(coord_y: np.ndarray):
    """
    Returns the max and mins of and 1-D array
    :param coord_y: 1-D numpy array
    :return: local maxima's, minimal
    """

    maxima = argrelextrema(np.array(coord_y), np.greater, order=2)
    minima = argrelextrema(np.array(coord_y), np.less, order=2)

    peaks = np.concatenate([maxima[0], minima[0]])
    peaks_s = np.sort(peaks)
    try:
        idx = [((peaks_s[sorted_p] + peaks_s[sorted_p - 1])//2) for sorted_p in range(1, len(peaks_s)-1)]
    except ValueError as v:

        raise v
    print(idx)
    return np.array(idx)


def eval_gradient(y_cords, diff_color=False):
    """
    Method that returns the number of events detected from input (y_cords, time) plot, by searching the peaks by use of
    gradient difference between neighbor predictions, and check if that difference is under a certain threshold.
    :param y_cords: ndarray with y-coordinates annotations (either from prediction or ground truth).
    :param diff_color: boolean indicating if show each event in one color.
    """
    x = np.arange(len(y_cords))
    step = 1 / len(x)
    print('number of frames: ', len(y_cords))
    grad_1 = np.gradient(y_cords, step)
    grad1_z = [grad_1 == 0]

    norm = np.linalg.norm(grad_1)

    grad1_norm = grad_1 / norm
    grad_signs = np.sign(grad1_norm)
    peaks = np.diff(grad_signs)
    peaks_values = np.nonzero(peaks)
    peaks_values = peaks_values[0]
    plt.figure()
    plt.xlim([0, x[-1]])
    plt.ylim([0, max(y_cords) + 20])

    if diff_color:
        color_up = 'red'
        color_down = 'green'
        for p in range(0, len(peaks_values)-2):
            curr_color = color_up
            if (y_cords[peaks_values[p+1]] - y_cords[peaks_values[p]]) > 50:
                curr_color = color_up
            if (y_cords[peaks_values[p+1]] - y_cords[peaks_values[p]]) < -50:
                curr_color = color_down
            plt.plot(np.arange(peaks_values[p], peaks_values[p+2]), y_cords[peaks_values[p]:peaks_values[p+2]],
                     color=curr_color)
    else:
        plt.plot(x, y_cords)

    plt.scatter(x[peaks_values], y_cords[peaks_values], marker='x', color='b')
    plt.scatter(x[grad1_z], y_cords[grad1_z], marker='x', color='b')
    plt.show()

    print('Number of events: ', len(peaks_values))
    return np.array(peaks_values)


def filter_down_peaks(f):
    """
    Method that smooths the predicted y-coordinates in order to avoid jumps in the graph when we do not have
    available predictions.
    :param f: ndarray with values to be filtered (in our case, predicted y-coordinates).
    :return: ndarray with smoothed values (avoiding jumps when no data is available -> avoiding False Positives
    detections in 'eval_gradient() function
    """
    thr = 200   # Threshold that decides if intra-frame y-coordinates differences is reasonable, if not, smooth value.
    for p in range(6, len(f)-1,):
        if abs(f[p]-f[p+1]) > thr:
            f[p+1] = f[p]
    return f


class BallTracking_improved:

    def __init__(self, video_name, Cfg, debug=False):

        self.name = None
        self.video_name = video_name
        self.video_capture = None
        self.video_writer = None
        self.v_height = None
        self.v_width = None
        self.v_num_frames = None
        self.v_fps = None
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self.predict_csv_file = None
        self.output_csv_file = None

        self.tracking_gt_path = None
        self.predict_path = None

        self.gt_annotations = []
        self.predictions = []
        self.custom_predictions = []

        self.debug = debug
        self.save_path = None
        self.input_path = None
        self.code_config = Cfg
        self.load()

    def load(self):

        # Define video name in class and related paths
        self.video_name = os.path.join(self.code_config.get_INPUT_VIDEO_DIR(), self.video_name)
        self.name = self.video_name.split('.mp4')[0]

        self.input_path = os.path.join(self.code_config.get_INPUT_VIDEO_DIR(), self.video_name)
        self.save_path = os.path.join(self.code_config.get_RESULT_VIDEO_OUTPUT(), 'custom_ball_Tracking.mp4')

        # load ground truth.
        self.tracking_gt_path = os.path.join(self.code_config.get_RESULT_VIDEO_OUTPUT(), 'Label.csv')
        self.predict_path = os.path.join(self.code_config.get_RESULT_VIDEO_OUTPUT(), 'ball_tracking_predict.csv')

        # Read input video
        self.video_capture = cv2.VideoCapture(self.video_name)

        self.v_fps, self.v_num_frames, self.v_width, self.v_height = get_video_properties(video=self.video_capture)

        # define codec and create VideoWriter object
        self.video_writer = cv2.VideoWriter(self.save_path, self.fourcc, self.v_fps,
                                            (self.v_width, self.v_height), True)

        result_path = os.path.join(self.code_config.get_RESULT_VIDEO_OUTPUT(), 'custom_ball_tracking.csv')
        self.output_csv_file = open(result_path, 'w')
        self.output_csv_file.write('Frame,Visibility,X,Y,Time\n')

    def custom_tracking(self):

        # Load Predicted values.
        self.load_predictions()

        try:
            # Load Ground truth values
            self.load_gt_annotations()
        except FileNotFoundError:
            # No GT available for this video! proceed without any gt annotations
            self.gt_annotations = self.predictions

        # Get metrics comparing GT and Vanilla TrackNet.
        get_metrics(gt=self.gt_annotations, prediction=self.predictions)

        # self.infer_parabolic_curve()
        self.custom_predictions = self.infer_trajectory_kalman()
        get_metrics(gt=self.predictions, prediction=self.custom_predictions)

        y_pred = get_y_cords(self.custom_predictions)
        eval_gradient(y_pred)

        y_filter = filter_down_peaks(y_pred)
        eval_gradient(y_filter, True)
        self.show_predictions(predictions_show=self.predictions)
        self.show_trajectory()

    def load_gt_annotations(self):
        """
        Function that will read and return ground truth annotation data (from the video evaluated with TrackNetV2)
        :return: list with all annotations.
        """

        # load Ground Truth annotations
        with open(self.tracking_gt_path, 'r') as label_obj:
            reader = csv.reader(label_obj)
            for row in reader:
                self.gt_annotations.append(label_instance(row[0], row[1], row[2], row[3], row[4]))

        # pop out first row (header)
        _ = self.gt_annotations.pop(0)

    def load_predictions(self):
        """
        Function that will read and return predicted annotation data (from inference done with TrackNetV2)
        :return: list with all annotations.
        """

        # load predictions from csv annotations file.
        with open(self.predict_path, 'r') as pred_obj:
            reader = csv.reader(pred_obj)
            for row in reader:
                self.predictions.append(label_instance(row[0], row[1], row[2], row[3], row[4]))

        # pop out first row (header)
        _ = self.predictions.pop(0)

    def infer_trajectory_kalman(self, mx=25, my=10, num_repressors=20):
        """
        Method that will infer missing data from predictions given by TrackNetV2 using Kalman filter.
        :param num_repressors: number previous points taken into account to predict next position
        :param mx: maximum x-axis distance allowed (to consider outlier)
        :param my: maximum y-axis distance allowed (to consider outlier)
        :return: list with predictions with missing data inferred.
        """
        # First, detect outliers from predicted instances
        predictions_list_, _ = outliers_detector(self.predictions, max_d_x=mx, max_d_y=my)
        self.custom_predictions = predictions_list_

        # count how many missing data we have to infer
        missing = []
        [missing.append(p) for p in predictions_list_ if int(p.vis) != 1]

        # extract all coordinates available
        cords = []
        [cords.append([p.x, p.y]) for p in self.predictions]

        # Initialize kalman Filter
        kf = KalmanFilter()

        # Iterate over predictions and infer missing data.
        for miss in range(len(missing)):

            miss_idx = self.custom_predictions.index(missing[miss])
            _num_regressors = min(miss_idx, num_repressors)
            k_start = miss_idx - _num_regressors
            kalman_argument = cords[k_start:miss_idx]
            kalman_valid = np.array(kalman_argument)

            # get future available data
            _prd = self.custom_predictions[miss_idx:]
            valid = []
            [valid.append(p) for p in _prd if int(p.vis) == 1]

            # get previous available data
            _cords = list(cords[:miss_idx])
            _pdr_prev = self.custom_predictions[:miss_idx]
            prev_valid = []
            [prev_valid.append(p) for p in _pdr_prev if int(p.vis) == 1]

            if kalman_argument:
                # predict coordinates with Kalman filter (using batch)
                kalman_p = [kf.predict(k_f_p[0], k_f_p[1]) for k_f_p in kalman_valid]
                kalman = kalman_p[-1]
                dif_x, dif_y = kalman[0] - _pdr_prev[-1].x, kalman[1] - _pdr_prev[-1].y

                # if predicted kalman coordinate is greater than allowed intra frame distance -> adapt prediction.
                if (abs(dif_x) > mx or abs(dif_y) > my) and dif_x > 0. and dif_y > 0.:

                    sign_x = dif_x / abs(dif_x)
                    sign_y = dif_y / abs(dif_y)
                    rectified_kalman = (_pdr_prev[-1].x + int(min(abs(dif_x), mx) * sign_x),
                                        _pdr_prev[-1].y + int(min(abs(dif_y), my) * sign_y))
                    self.custom_predictions[miss_idx].x = abs(rectified_kalman[0])
                    self.custom_predictions[miss_idx].y = abs(rectified_kalman[1])
                else:
                    self.custom_predictions[miss_idx].x = abs(kalman[0])
                    self.custom_predictions[miss_idx].y = abs(kalman[1])

                self.custom_predictions[miss_idx].vis = 1
                self.custom_predictions[miss_idx].status = 'kalman'
                cords[miss_idx] = [abs(self.custom_predictions[miss_idx].x), abs(self.custom_predictions[miss_idx].y)]
            else:
                # Get next valid position as current if no previous data available to perform kalman prediction
                self.custom_predictions[miss_idx].x = valid[0].x
                self.custom_predictions[miss_idx].y = valid[0].y
                self.custom_predictions[miss_idx].vis = valid[0].vis
                cords[miss_idx] = cords[miss_idx + 1]

        return self.custom_predictions

    def infer_parabolic_curve(self):

        def parabola(coef, x):
            a, b, c = coef
            return a * x ** 2 + b * x + c

        def residuals(coef, x, y):
            return parabola(coef, x) - y

        # Get ball position data over time.
        time = np.arange(0, self.v_num_frames-1, 1)
        position_x = [0] * (self.v_num_frames - 1)
        position_y = [0] * (self.v_num_frames - 1)

        positions_initi = self.predictions
        for i, p in enumerate(positions_initi):
            position_x[i] = p.x
            position_y[i] = p.y

        # Approximate initial coefficients of the parabolic equation (y = a(x²) + b(x) + c
        initial_guess = [1, 1, 1]

        # Get detected positions of TrackNet (badminton)
        position_x = np.array(position_x)
        position_y = np.array(position_y)
        time = np.array(time)

        # Use 'least squares' to adjusts samples over time (ball position) to the parabolic curve form.
        result = least_squares(residuals, initial_guess, args=(time, position_y))
        a_fit, b_fit, c_fit = result.x

        # load video to show results.
        video_path = self.video_name
        cap = cv2.VideoCapture(video_path)

        ret, frame = cap.read()
        height, width, _ = frame.shape

        # Create output video writer.
        output_path = 'video_salida.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Compute ball coordinates with parabolic correction.
            x_current = frame_num
            y_current = parabola([a_fit, b_fit, c_fit], x_current)

            x_img = position_x[frame_num - 1]
            y_img = position_y[frame_num - 1]

            cv2.circle(frame, (x_img, y_img), 5, (0, 0, 255), -1)
            cv2.line(frame, (x_img, y_img), (int(x_current), int(y_current)), (0, 255, 0), 2)

            cv2.imshow('parabolic adjustment', frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def show_trajectory(self):
        """
        Function that plot in a 3D plot, the ball trajectory given.
        :return:
        """
        fig = plt.figure()
        z = np.arange(len(self.gt_annotations))

        ay = fig.add_subplot(131, projection='3d')
        x = []
        y = []
        [x.append(npp.x) for npp in self.gt_annotations]
        [y.append(npp.y) for npp in self.gt_annotations]

        ay.set_xlabel('time')
        ay.set_title('Ground Truth trajectory')
        ay.set_ylabel('x-coord')
        ay.set_zlabel('y-coord')
        ay.plot(z, np.array(x), np.array(y))

        ax = fig.add_subplot(133, projection='3d')
        x = []
        y = []
        [x.append(npp.x) for npp in self.custom_predictions if npp.vis == 1 and (npp.x != 0 and npp.y != 0)]
        [y.append(npp.y) for npp in self.custom_predictions if npp.vis == 1 and (npp.x != 0 and npp.y != 0)]
        z_ = np.arange(len(y))

        # Set common labels
        ax.set_xlabel('time')
        ax.set_ylabel('x-coord')
        ax.set_zlabel('y-coord')

        ax.set_title('TrackNet + data inference')
        ax.plot(z_, np.array(x), np.array(y))

        az = fig.add_subplot(132, projection='3d')
        x = []
        y = []
        [x.append(npp.x) for npp in self.predictions if npp.vis == 1 and (npp.x != 0 and npp.y != 0)]
        [y.append(npp.y) for npp in self.predictions if npp.vis == 1 and (npp.x != 0 and npp.y != 0)]
        z_ = np.arange(len(y))

        az.set_xlabel('time')
        az.set_title('TrackNet prediction')
        az.set_ylabel('x-coord')
        az.set_zlabel('y-coord')
        az.plot(z_, np.array(x), np.array(y))

        plt.show()

    def show_predictions(self, predictions_show):

        cap = self.video_capture
        number_frames = self.v_num_frames

        for i in range(number_frames):
            success, image = cap.read()

            wk = 100

            if predictions_show[i].vis == 1:
                if predictions_show[i].status == 'kalman':
                    color = (255, 0, 0)
                    wk = 200
                else:
                    color = (0, 0, 255)
                cv2.circle(image, (predictions_show[i].x, predictions_show[i].y), 5, color, -1)
                if predictions_show[i - 1] and predictions_show[i - 1].vis == 1:
                    cv2.arrowedLine(image, (predictions_show[i - 1].x, predictions_show[i - 1].y),
                                    (predictions_show[i].x, predictions_show[i].y), (0, 0, 0), thickness=5)

            cv2.imshow('video', image)
            cv2.waitKey(wk)
            self.video_writer.write(image)

        cv2.destroyAllWindows()
        self.release_files()

    def release_files(self):
        # self.predict_csv_file.close()
        self.output_csv_file.close()
        self.video_capture.release()
        self.video_capture.release()
