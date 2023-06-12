import numpy as np
import cv2


class KalmanFilter:

    def __init__(self):
        pass

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def predict(self, x_cord, y_cord):
        """
        Function used to estimate next position given previous data by using kalman filter.
        :param x_cord: list corresponding to x-coordinates
        :param y_cord: list corresponding to y-coordinates
        :return: estimation of next point.
        """
        a = np.float32(x_cord)
        b = np.float32(y_cord)
        measured = np.array([[a], [b]])

        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y
