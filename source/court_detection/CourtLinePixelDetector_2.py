import numpy as np
import warnings
import cv2

warnings.filterwarnings('error')


class CourtLinePixelDetector:

    def __init__(self, debug=False, grayscale_min=190, neighborhood_diff=30, court_line_width=2, grad_kernel_size=3,
                 kernel_size=3):
        """
        Initialization function for CourtLinePixelDetector class
        :param debug: boolean variable to show (True) or not (False) the different stages in line detection.
        :param luminance_min: minimum value to consider a pixel as court line pixel (in luminance image)
        :param neighborhood_diff: maximum difference to consider pixel as part of a line structure.
        :param court_line_width: court line pixel width
        :param grad_kernel_size: Size of kernel to compute gradient (via Sobel operators)
        :param kernel_size: Size of kernel to compute Gaussian Blurr.
        """

        # Initialization of default values:
        self.debug = debug
        self.grayscaleThr = grayscale_min
        self.darkThr = neighborhood_diff
        self.theta = court_line_width
        self.gradientKernelSize = grad_kernel_size
        self.kernelSize = kernel_size

        # Define parameters to be used in the class:
        self.frame = None
        self.width = None
        self.height = None
        self.gray = None
        self.bw_image = None
        self.image = None
        self.court_line_pixels = None

    def CourtLinePixelDetector(self, frame):
        """
        Function that executes all class's function in order to return a binary image (white pixels: court candidate)
        :param frame: input video's frame from which detect line
        :return court_line_pixels modified (where white pixels are the ones detected as court line)
        """
        # Define class variables:
        self.frame = frame
        self.width, self.height, _ = self.frame.shape
        w = self.width
        h = self.height

        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.gray = np.zeros((w, h))
        self.gray[w // 8:7 * w // 8, h // 8:7 * h // 8] = gray[w // 8:7 * w // 8, h // 8:7 * h // 8]

        bw = cv2.threshold(self.gray, self.grayscaleThr, 255, cv2.THRESH_BINARY)[1]

        cv2.imshow('black and white image', bw)
        cv2.waitKey(0)

        # Compute line detection (fill court_line_pixels's variable).
        self.court_line_pixels = np.zeros((self.width, self.height))
        self.court_line_pixels = self.detectLinePixels()

        if self.debug:
            cv2.imshow('final_result', self.court_line_pixels)
            if cv2.waitKey(0) & 0xff == 27:
                cv2.destroyAllWindows()

        return self.court_line_pixels

    def getLuminanceChannel(self):
        """
        Extract luminance channel using OpenCv's color space transformation.
        Note! Showing both input frame and its luminance channel if debug is True.
        :return: luminance_img ()
        """
        luminance_img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)[..., 2]
        if self.debug:
            cv2.imshow('Original image: ', self.frame)
            cv2.imshow('Luminance channel: ', self.luminance_img)
            cv2.waitKey(0)
        return luminance_img

    def detectLinePixels(self):
        """
        Function used to detect which pixels belongs to a line
        :return: pixelImage (np.matrix) with same shape as input image but only with those pixels that actually belongs
        to a line.
        """
        # Define mask where we'll store the line pixels (with same size as input frame)
        court_line_pixels = np.zeros((self.width, self.height))

        # Iterate over image
        for x in range(0, self.width):
            for y in range(0, self.height):

                # Define list to consider a a pixel part of a line structure
                court_line_pixels[x, y] = self.isLinePixel(x, y)

        return court_line_pixels

    def isLinePixel(self, x: int, y: int):
        """
        Function that checks if selected pixels belongs to a line, comparing the 'theta' horizontal ande vertical
        neighboring pixels and check if difference between current pixl value, and it's theta neighbors is below
        'dark_thr'

        :param tmp_theta: amount of neighboring (both vertical and horizontal) pixels to consider, depends on court line
        width defined in initialization
        :param x: position x in loop
        :param y: position y in loop
        :return: False if pixel does not belong to line, True otw.
        """

        value = self.gray[x, y]

        if value < self.grayscaleThr:
            return False
        # Return False if pixel evaluated is in boundaries of image.
        if (x < self.theta or self.width - x <= self.theta) or (y < self.theta or self.height - y <= self.theta):
            return False

        # Compute vertical and horizontal differences
        dv1 = abs(value - int(self.gray[x, y - self.theta]))
        dv2 = abs(value - int(self.gray[x, y + self.theta]))
        dh1 = abs(value - int(self.gray[x - self.theta, y]))
        dh2 = abs(value - int(self.gray[x + self.theta, y]))

        try:
            # Return True if difference between evaluated pixel and neighboring pixels is below darkThr
            if (dh1 > self.darkThr and dh2 > self.darkThr) or (dv1 > self.darkThr and dv2 > self.darkThr) \
                    and value > self.grayscaleThr:
                return True
            # Return False otherwise
            else:
                return False

        # Added for debugging purposes (and to avoid code crashing)
        except RuntimeWarning as e:
            print('Value: ', value)
            print('left Value: ', dh1)
            print('right Value: ', dh1)
            print(' Difference: ', self.darkThr)
            print(e)
            return False

    def filterLinePixels(self):
        """
        NOTE: CURRENTLY NOT BEING USED DUE TO BAD PERFORMANCE!
        Function that filters out regions with high texture using the second moment matrix (uses First and second order
        derivatives, that is Sobel and Hessian operators).
        Note! High computational cost and poor performance in terms of discarding high texture regions (or corners)
        Goal was to extract the eigenvalues from the second moment matrix and keep those regions where that eigenvalues
        indicate that we are in a edge (or as we are interested, a line).
        :return: court_line_pixels modified with the
        """
        luminance_float = np.copy(self.luminance_img)
        luminance_float.astype(np.float32)

        blurred_luminance = cv2.GaussianBlur(luminance_float, (self.kernelSize, self.kernelSize), 0)
        dx = cv2.Sobel(blurred_luminance, cv2.CV_64F, 1, 0, ksize=self.gradientKernelSize)
        dy = cv2.Sobel(blurred_luminance, cv2.CV_64F, 0, 1, ksize=self.gradientKernelSize)

        dx2, dxy, dy2 = self.computeStructureTensorElements(dx, dy)

        binaryImage = np.copy(self.luminance_img)
        for x in range(self.width):
            for y in range(self.height):
                value = binaryImage[x, y]
                if value != 0 and luminance_float[x, y] > self.lumThr:
                    t = np.empty((2, 2), np.float32)
                    t[0, 0] = dx2[x, y]
                    t[0, 1] = dxy[x, y]
                    t[1, 0] = dxy[x, y]
                    t[1, 1] = dy2[x, y]

                    l = cv2.eigenNonSymmetric(t)
                    w = l[0]
                    # w, _ = np.linalg.eig(t)
                    try:
                        if abs(w[0]) > (100 * abs(w[1])):
                            self.court_line_pixels[x, y] = 255
                    except RuntimeWarning as R:
                        print(R)
                        print('EigenValues: ', w)
                        self.court_line_pixels[x, y] = 0

    def computeStructureTensorElements(self, dx, dy):
        """
        NOTE: CURRENTLY NOT BEING USED DUE TO BAD PERFORMANCE! (as it is only being used in filterLinePixels function)
        Function that returns second order derivatives given first order derivatives (goal is to extract the Hessian
        matrix of a patch surrounding the evaluated pixel, to the extract the eigenvalues from the second moment matrix)

        :param dx: Gradient in x-axis (from sobel operator)
        :param dy: Gradient in y-axis (from sobel operator)
        :return: dx2, dxy, dy2: second order derivatives (to construct Hessian matrix)
        """
        dx2 = cv2.multiply(dx, dx)
        dxy = cv2.multiply(dx, dy)
        dy2 = cv2.multiply(dy, dy)

        kernel = np.ones((self.kernelSize, self.kernelSize), np.float32)
        dx2 = cv2.GaussianBlur(dx2, (self.kernelSize, self.kernelSize), 0)
        dxy = cv2.GaussianBlur(dxy, (self.kernelSize, self.kernelSize), 0)
        dy2 = cv2.GaussianBlur(dy2, (self.kernelSize, self.kernelSize), 0)

        return dx2, dxy, dy2
