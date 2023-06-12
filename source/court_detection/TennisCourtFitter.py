from itertools import combinations

from source.court_detection.CourtLineCandidateDetector import line_intersection
from source.court_detection.CourtRefernce import CourtReference

import numpy as np
import cv2
import os


def sort_intersection_points(intersections):
    """
    sort intersection points from top left to bottom right
    """
    y_sorted = sorted(intersections, key=lambda x: x[1])
    p12 = y_sorted[:2]
    p34 = y_sorted[2:]
    p12 = sorted(p12, key=lambda x: x[0])
    p34 = sorted(p34, key=lambda x: x[0])
    return p12 + p34


def _threshold(frame):
    """
    Simple thresholding for white pixels
    To generate 'Correct' mask used in get_confi_score
    :return: gray (mask) with the brightest pixels (>190) as white pixels, and the rest as black pixels.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    return gray


class TennisCourtFitter:

    def __init__(self, debug=True):
        """
        Class initialization function
        :param debug: boolean variable to show (True) or not (False) the different stages in line detection.
        """

        # Define image dependent variables
        self.final_dir = None
        self.binary_mask = None
        self.gray = None
        self.vertical = None
        self.frame = None
        self.horizontal = None
        self.height = None
        self.width = None
        self.binaryImage = None
        self.candidate_lines = None

        # Define court variables
        self.baseline_top = None
        self.baseline_bottom = None
        self.net = None
        self.left_court_line = None
        self.right_court_line = None
        self.left_inner_line = None
        self.right_inner_line = None
        self.middle_line = None
        self.top_inner_line = None
        self.bottom_inner_line = None
        self.success_flag = False
        self.best_conf = None
        self.frame_points = None

        # Define and set rest of variables.
        self.court_reference = CourtReference()
        self.court_warp_matrix = []
        self.game_warp_matrix = []
        self.court_score = 0
        self.success_accuracy = 80
        self.success_score = 1000
        self.debug = debug
        self.refinement_iterations = 5

    def TennisCourtFitter(self, candidate_lines, binaryImage, frame, final_dir):
        """
        Function that executes all class's function in order to obtain a reference_court to frame correspondence.
        :param final_dir:
        :param candidate_lines: list with candidate lines from which obtain correspondences with reference court.
        :param binaryImage: average of 'n' masks obtained from 'courtLinePixelDetector()' class, mask of candidate pixel
        :param frame: RGB image
        :returns: self.find_lines_location() [list with all lines], court_warp_matrix [the best Homography matrix found]
        and game_warp_matrix [inverse of the best Homography matrix found].
        """
        self.final_dir = final_dir

        self.candidate_lines = candidate_lines
        self.binaryImage = binaryImage
        self.width, self.height = self.binaryImage.shape
        self.frame = frame

        self.gray = _threshold(self.frame)

        horizontal_lines = self.candidate_lines[0]
        vertical_lines = self.candidate_lines[1]

        gray = self.gray.copy()
        gray[gray > 0] = 1

        self.binary_mask = np.zeros((self.width, self.height))
        self.binary_mask[self.width // 8:7 * self.width // 8, self.height // 8:7 * self.height // 8] = \
            gray[self.width // 8:7 * self.width // 8, self.height // 8:7 * self.height // 8]

        if self.debug:
            cv2.imshow('Binary mask to compute score', self.binary_mask)
            cv2.waitKey(0)

        # Find transformation from reference court to frame`s court
        court_warp_matrix, game_warp_matrix = self._find_homography(horizontal_lines, vertical_lines)

        # Add matrices to class variables.
        self.court_warp_matrix.append(court_warp_matrix)
        self.game_warp_matrix.append(game_warp_matrix)

        return self.find_lines_location(), court_warp_matrix, game_warp_matrix

    def _find_homography(self, horizontal_lines, vertical_lines):
        """
        Finds transformation from reference court to frame`s court using 4 pairs of matching points
        :param horizontal_lines: list with clean and filtered horizontal lines
        :param vertical_lines: list with clean and filtered vertical lines
        :return: Homography matrix, inverse Homography matrix, score of candidate.
        """
        max_score = -np.inf
        max_mat = None
        max_inv_mat = None
        k = 0
        invalid_set = 0
        # Loop over every pair of horizontal lines and every pair of vertical lines
        for horizontal_pair in list(combinations(horizontal_lines, 2)):
            for vertical_pair in list(combinations(vertical_lines, 2)):
                h1, h2 = horizontal_pair
                v1, v2 = vertical_pair
                # Finding intersection points of all lines
                i1 = line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v1[0:2]), tuple(v1[2:])))
                i2 = line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v2[0:2]), tuple(v2[2:])))
                i3 = line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v1[0:2]), tuple(v1[2:])))
                i4 = line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v2[0:2]), tuple(v2[2:])))

                intersections = [i1, i2, i3, i4]
                intersections = sort_intersection_points(intersections)

                for i, configuration in self.court_reference.court_conf.items():
                    # Find transformation
                    matrix, _ = cv2.findHomography(np.float32(configuration), np.float32(intersections),
                                                   cv2.RANSAC, self.refinement_iterations)
                    inv_matrix = cv2.invert(matrix)[1]
                    # confi_score, invalid_set = self.check_score_homography(matrix, invalid_set, max_score)
                    confi_score = self._get_confi_score(matrix)

                    if max_score < confi_score:
                        max_score = confi_score
                        max_mat = matrix
                        max_inv_mat = inv_matrix
                        self.best_conf = i

                    k += 1

        print('numero de sets probados: ', k)
        print('numero de sets invalidos (no computed score): ', invalid_set)
        self._get_confi_score(max_mat)
        if self.debug:
            # Store and show reference court overlay to input rgb frame
            frame = self.frame.copy()
            court = self.add_court_overlay(frame, max_mat, (255, 0, 0))
            if cv2.waitKey(0) & 0xff == 27:
                cv2.destroyAllWindows()

            cv2.imwrite(os.path.join(self.final_dir, "court_over_frame.png"), court)
            cv2.imwrite(os.path.join(self.final_dir, "frame.png"), self.frame)

        self.court_score = max_score
        print(f'Score = {max_score}')
        print(f'Combinations tested = {k}')

        return max_mat, max_inv_mat

    def check_score_homography(self, matrix, invalid_set, max_score):
        f_square = - (matrix[0][0] * matrix[0][1] + matrix[1][0] * matrix[1][1]) / matrix[2][0] * matrix[2][1]
        beta = (matrix[0][1] ** 2 + matrix[1][1] ** 2 + f_square * matrix[2][1] ** 2) / \
               (matrix[0][0] ** 2 + matrix[1][0] ** 2 + f_square * matrix[2][0] ** 2)

        if beta > 4.0 or beta < 0.25:
            print('invalid_set')
            invalid_set += 1
            confi_score = max_score
        else:
            print(beta)
            # Get transformation score
            confi_score = self._get_confi_score(matrix)

        return confi_score, invalid_set

    def _get_confi_score(self, matrix):
        """
        Calculate transformation score, warping reference_court using Homography given as input and compare pixels with
        the brightest pixels from the input frame.
        :param matrix: Homography matrix.
        :return: score (indicates how good the homography evaluated is warping reference court to actual court)
        """

        # 1st: warp court reference using input's homography matrix and binarize to obtain a mask.
        court = cv2.warpPerspective(self.court_reference.court, matrix, self.frame.shape[1::-1])
        court[court > 0] = 1

        # 2nd: generate mask from input rgb frame taking only the brightest pixels

        # mask = cv2.bitwise_and(gray.astype('uint8'), self.binaryImage.astype('uint8'))
        # mask = gray * self.binaryImage
        # mask[mask > 0] = 1

        # 3rd: compute correct (multiplying gray_mask with court_mask) and incorrect (computing court - correct pixels)
        correct = court * self.binary_mask
        wrong = court - correct

        # 4th: sum all correct and incorrect pixels in each mask and compute score via: score = Correct - 1/2 * Wrong
        # Note that correct pixels add 1, and wrong pixels subtracts 0.5 to the score.
        overall_correct = np.sum(correct)
        overall_wrong = np.sum(wrong)
        score = overall_correct - 0.5 * overall_wrong

        return score

    def add_court_overlay(self, frame, homography=None, overlay_color=(255, 255, 255), frame_num=-1):
        """
        Add overlay of the court to the frame. Function used only used if debug=True (to show court over RGB frame)
        :param frame: RGB frame
        :param homography: (optional) Homography to use to warp reference court model onto RGB image.
        :param overlay_color: color of court lines (white by default)
        :param frame_num: index of homography to get from court_warp_matrix list (by default take last one added)
        :return: frame (with reference court mask over RGB input frame).
        """
        if homography is None and len(self.court_warp_matrix) > 0 and frame_num < len(self.court_warp_matrix):
            homography = self.court_warp_matrix[frame_num]
        court = cv2.warpPerspective(self.court_reference.court, homography, frame.shape[1::-1])
        frame[court > 0, :] = overlay_color
        return frame

    def find_lines_location(self):
        """
        Finds important lines location on frame.
        Useful to generate accuracy metrics (knowing ground truth).
        :return: lines (list) with all lines.

        Note reference Lines are structured as:

        baseline_top = lines[:4]
        baseline_bottom = lines[4:8]
        net = lines[8:12]
        left_court_line = lines[12:16]
        right_court_line = lines[16:20]
        left_inner_line = lines[20:24]
        right_inner_line = lines[24:28]
        middle_line = lines[28:32]
        top_inner_line = lines[32:36]
        bottom_inner_line = lines[36:40]
        """

        p = np.array(self.court_reference.get_important_lines(), dtype=np.float32).reshape((-1, 1, 2))
        lines = cv2.perspectiveTransform(p, self.court_warp_matrix[-1]).reshape(-1)

        return lines
