import numpy as np
import cv2
from source.court_detection.line import Line
from sympy import Line


def line_intersection(line1, line2):
    """
    Find 2 lines intersection point
    """
    l1 = Line(line1[0], line1[1])
    l2 = Line(line2[0], line2[1])

    intersection = l1.intersection(l2)
    return intersection[0].coordinates


def display_lines_on_frame(frame, horizontal=(), vertical=()):
    """
    Display lines on frame for horizontal and vertical li
                nes
    """
    for line in horizontal:
        x1, y1, x2, y2 = line.astype(np.uint)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 5)
        cv2.circle(frame, (x1, y1), 1, (255, 0, 0), 2)
        cv2.circle(frame, (x2, y2), 1, (255, 0, 0), 2)

    for line in vertical:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 5)
        cv2.circle(frame, (x1, y1), 1, (255, 0, 0), 2)
        cv2.circle(frame, (x2, y2), 1, (255, 0, 0), 2)

    cv2.imshow('Detected lines', frame)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    return frame


class CourtLineCandidateDetector:

    def __init__(self, debug=True, houghThreshold=100, distanceThreshold=35):
        """
        Class initialization function
        :param debug: boolean variable to show (True) or not (False) the different stages in line detection.
        :param houghThreshold: minimum voting value to consider a line in the OpenCv Hough Lines Probabilistic function.
        :param distanceThreshold: maximum distance to consider two lines as the same within the HoughLinesP function.
        """
        self.height = None
        self.width = None
        self.frame = None
        self.binaryImage = None

        # Define class specific parameters
        self.lines = []
        self.houghThreshold = houghThreshold
        self.distanceThreshold = distanceThreshold
        self.debug = debug
        self.pixel_margin = 5

    def CourtLineCandidateDetector(self, binaryImage, frame):
        """
        Function that executes all class's function in order to return two lists: horizontal and vertical lines
        :param binaryImage: average mask of candidate pixels using 'n' consecutive frames
        :param frame: RGB image (first frame to be considered)
        :return: horizontal (list) and vertical (list)
        """
        self.binaryImage = binaryImage
        self.frame = frame
        self.width, self.height = self.binaryImage.shape

        horizontal, vertical = self.extractLines()

        # Merge lines that belong to the same line on frame
        horizontal, vertical = self._merge_lines(horizontal, vertical)
        # vertical = self.filter_v_lines(vertical)
        horizontal = self.filter_h_lines(vertical, horizontal)

        if self.debug:
            display_lines_on_frame(frame.copy(), horizontal, vertical)

        return horizontal, vertical

    def extractLines(self):
        """
        Function that extract lines using the HoughLines Probabilistic function from OpenCV, it receives as input a
        filtered binaryImage obtained from courtLinePixelDetector's class.
        :return: horizontal (list) and vertical (list) lines if lines were detected with the Hough Transform or two
        empty lists in case of nothing was detected.
        """
        tmpLines = cv2.HoughLinesP(self.binaryImage.astype(np.uint8), 1, np.pi / 180,
                                   self.houghThreshold, minLineLength=100, maxLineGap=15)
        tmpLines = np.squeeze(tmpLines)

        img = self.frame.copy()
        color = (0, 255, 0)
        print(len(tmpLines), ' lines detected')
        if tmpLines is not None:
            for line in tmpLines:
                x1, y1, x2, y2 = line
                l = np.array((x1, y1, x2, y2))
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
                self.lines.append(l)

            if self.debug:
                cv2.imshow('Candidate lines detected', img)
                cv2.waitKey(0)

            horizontal, vertical = self._classify_lines()
            return horizontal, vertical
        else:
            return [], []

    def _classify_lines(self):
        """
        Classify line to vertical and horizontal lines.
        Using [dx = p1.x - p2.x] and [dy = p1.y - p2.y], if x-axis distance is grater than double of y-axis distance,
        set evaluated line as horizontal, and set as vertical otherwise.
        Additionally, clearing horizontal lines (setting that no horizontal line considered as candidate if it's outside
        the range of vertical lines, i.e.: if some horizontal line starts and ends in a position of a image that is out
        of range of any vertical line, delete evaluated horizontal line in clean_horizontal list.)
        :return: clear_horizontal, vertical (if lines were detected from Hough Transform) or two empty lists if not.
        """
        horizontal = []
        vertical = []
        highest_vertical_y = np.inf
        lowest_vertical_y = 0

        if self.lines is not None:

            for line in self.lines:
                x1, y1, x2, y2 = line

                p1 = np.array((x1, y1))
                p2 = np.array((x2, y2))

                dx = abs(p1[0] - p2[0])
                dy = abs(p1[1] - p2[1])

                if dx > 2 * dy:
                    horizontal.append(line)
                else:
                    vertical.append(line)
                    highest_vertical_y = min(highest_vertical_y, y1, y2)
                    lowest_vertical_y = max(lowest_vertical_y, y1, y2)

            # Filter horizontal lines using vertical lines lowest and highest point
            clean_horizontal = []
            h = lowest_vertical_y - highest_vertical_y
            lowest_vertical_y += h / 15
            highest_vertical_y -= h * 2 / 15
            for line in horizontal:
                x1, y1, x2, y2 = line
                if lowest_vertical_y > y1 > highest_vertical_y and lowest_vertical_y > y1 > highest_vertical_y:
                    clean_horizontal.append(line)
            return clean_horizontal, vertical

        else:
            print('No lines detected!')
            return [], []

    def _merge_lines(self, horizontal_lines, vertical_lines):
        """
        Merge lines that belongs to the same frame`s lines
        :param: horizontal_lines: list of horizontal lines (derived from _classify_lines function)
        :param: vertical_lines: list of vertical lines (derived from _classify_lines function)
        :return: new_horizontal_lines and new_vertical_lines: list of final lines (once nearby lines are merged)
        """

        # Merge horizontal lines
        horizontal_lines = sorted(horizontal_lines, key=lambda item: item[0])
        mask = [True] * len(horizontal_lines)
        new_horizontal_lines = []
        for i, line in enumerate(horizontal_lines):
            if mask[i]:
                for j, s_line in enumerate(horizontal_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        dy = abs(y3 - y2)
                        if dy < 10:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[0])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False
                new_horizontal_lines.append(line)

        # Merge vertical lines
        vertical_lines = sorted(vertical_lines, key=lambda item: item[1])
        xl, yl, xr, yr = (0, self.height * 6 / 7, self.width, self.height * 6 / 7)
        mask = [True] * len(vertical_lines)
        new_vertical_lines = []
        for i, line in enumerate(vertical_lines):
            if mask[i]:
                for j, s_line in enumerate(vertical_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        xi, yi = line_intersection(((x1, y1), (x2, y2)), ((xl, yl), (xr, yr)))
                        xj, yj = line_intersection(((x3, y3), (x4, y4)), ((xl, yl), (xr, yr)))

                        dx = abs(xi - xj)
                        if dx < 10:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[1])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False

                new_vertical_lines.append(line)

        return new_horizontal_lines, new_vertical_lines

    def filter_v_lines(self, v_lines):
        """
        function that filters out outlier vertical lines, based on it's position within the frame
        :param v_lines: list with all vertical lines
        :return: clean vertical lines
        """
        right_vertical, left_vertical = [], []
        for line in v_lines:
            x1, _, x2, _ = line
            dist_x1 = x1 - self.width // 2
            dist_x2 = x2 - self.width // 2

            min(dist_x1, dist_x2)
            min(abs(dist_x1), abs(dist_x2))
            dist = [min(abs(dist_x1), abs(dist_x2)) / dist_x1, min(abs(dist_x1), abs(dist_x2)) / dist_x2]
            arr = np.array(dist)
            idx = np.where(arr.astype(int) == arr)[0][0]

            distances = np.array((dist_x1, dist_x2))
            closer_to_center = distances[idx]

            if closer_to_center > 0.:
                right_vertical.append(line)
            else:
                left_vertical.append(line)

        print('right and left lines: ', len(right_vertical), ' , ', len(left_vertical))
        clean_v_lines = []

        for r_line in right_vertical:
            x1, y1, x2, y2 = r_line
            p1 = np.array((x1, y1))
            p2 = np.array((x2, y2))
            max_y = max(y1, y2)
            if np.isin(max_y, p2).any():
                if p2[1] >= p1[1]:
                    clean_v_lines.append(r_line)
            else:
                if p1[1] >= p2[1]:
                    clean_v_lines.append(r_line)

        for l_line in left_vertical:
            x1, y1, x2, y2 = l_line
            p1 = np.array((x1, y1))
            p2 = np.array((x2, y2))
            max_y = max(y1, y2)
            if np.isin(max_y, p2).any():
                if p1[1] >= p2[1]:
                    clean_v_lines.append(l_line)
            else:
                if p2[1] >= p1[1]:
                    clean_v_lines.append(l_line)

        print('Input vertical lines: ', len(v_lines), ' clean vertical lines: ', len(clean_v_lines))
        return clean_v_lines

    def filter_h_lines(self, v_lines, h_lines):
        """
        Function that filter out horizontal lines with y-coordinates outside range defined by vertical lines.
        :param v_lines: List of vertical merged lines
        :param h_lines: List of horizontal merged lines
        :return: h_clean_lines: Filtered list of horizontal lines.
        """
        # Start with definition of minimum and maximum y-cords (for vertical lines only).
        min_v_x = np.inf
        max_v_x = 0

        min_v_y = np.inf
        max_v_y = 0

        for v_line in v_lines:
            x1, y1, x2, y2 = v_line
            if min_v_y > min(y1, y2):
                min_v_y = min(y1, y2)
            if max_v_y < max(y1, y2):
                max_v_y = max(y1, y2)
            if min_v_x > min(x1, x2):
                min_v_x = min(x1, x2)
            if max_v_x < max(x1, x2):
                max_v_x = max(x1, x2)

        # Add 'pixel_margin' (5) # of pixels as margin
        min_v_y -= self.pixel_margin
        max_v_y += self.pixel_margin
        min_v_x -= self.pixel_margin
        max_v_x += self.pixel_margin

        # Only return horizontal lines within vertical lines margins.
        new_h = []
        for h_line in h_lines:

            xh1, yh1, xh2, yh2 = h_line

            # checking if horizontal line is within vertical lines margins.
            check_y1 = min_v_y <= yh1 <= max_v_y
            check_y2 = min_v_y <= yh2 <= max_v_y
            check_x1 = min_v_x <= xh1 <= max_v_x
            check_x2 = min_v_x <= xh2 <= max_v_x

            if check_y1 and check_y2 and check_x1 and check_x2:
                new_h.append(h_line)

        return new_h
