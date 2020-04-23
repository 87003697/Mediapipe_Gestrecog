import cv2
import os
from hand_tracker import HandTracker
import numpy as np
import math


class Extractor_v2:
    """
    class extracting angles within keypoints detected by hand_  from raw images
    N
    Arg:
        image: raw image data from webcamera or dataset
    Outputs:
        (15,1) array of keypoints angles.
        keypoints detected
    """

    def __init__(self):
        # Initializing the HandTracker #################################################
        palm_model_path = "./models/palm_detection.tflite"
        landmark_model_path = "./models/hand_landmark.tflite"
        anchors_path = "./data/anchors.csv"
        self.detector = HandTracker(palm_model_path, landmark_model_path, anchors_path,
                                    box_shift=0.1, box_enlarge=1.5)
        self.kp_single_int = tuple([0, 0])  # initialized as the result

        # Setting keypoints pairs ###############################
        # self.pair_WRONG = [[1, 2], [2, 3], [3, 4],
        #                    [5, 6], [6, 7], [7, 8],
        #                    [9, 10], [10, 11], [11, 12],
        #                    [13, 14], [14, 15], [15, 16],
        #                    [17, 18], [18, 19],
        #                    [19, 20]]  # Adding a WRONG at the end of function name just means this is a
        # # wrong way of calculation
        self.pair = [[0, 1, 2], [1, 2, 3], [2, 3, 4],
                     [0, 5, 6], [5, 6, 7], [6, 7, 8],
                     [0, 9, 10], [9, 10, 11], [10, 11, 12],
                     [0, 13, 14], [13, 14, 15], [14, 15, 16],
                     [0, 17, 18], [17, 18, 19], [18, 19, 20]]

        self.stem_pair = [0, 9]  # palm points and root middle-finger point

    # @staticmethod
    # def _angle_WRONG(Y, X):
    #     # given cords of y and x in tuple format, returns arctan y x
    #     """
    #     Adding a WRONG at the end of function name just means this is a wrong way of calculation
    #     :param Y: cords of point 1
    #     :param X: cords of point 2
    #     :return:
    #     """
    #     y1 = Y[1]
    #     y2 = X[1]
    #     x1 = Y[0]
    #     x2 = X[0]
    #     arctan = math.atan2(y1 - y2, x1 - x2)
    #     return arctan

    @staticmethod
    def _angle(p2, p1, p3):
        """
        calculating the angle given three cords of point
        :param p1: cord of point on the angle
        :param p2: cord of point next to the angle
        :param p3: cord of another point next to the angle
        :return: the angles ranging from -1 to 1
        """
        a = (p2[0] - p3[0]) * (p2[0] - p3[0]) + (p2[1] - p3[1]) * (p2[1] - p3[1])
        b = (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])
        c = (p1[0] - p3[0]) * (p1[0] - p3[0]) + (p1[1] - p3[1]) * (p1[1] - p3[1])
        A = math.degrees(math.acos((b + c - a) / (2 * math.sqrt(b * c))))
        # A = A / 180 # This is for a number normalization

        return A

    def run(self, image):
        """

        :param image: image from which skeletons are detected
        :return: kp_angle: the angle data for classification
        """

        # Calibrating Channels ################################
        b, g, r = cv2.split(image)
        image_cali = cv2.merge([r, g, b])
        # Or using image_calis = image[:,:,::-1]

        # Extracting keypoints #######################################
        kp, box = self.detector(image_cali)

        if type(kp) is np.ndarray:
            kp_tuple = {}
            i = 0
            for kp_single in kp:
                kp_single_int = tuple([int(kp_single[0]), int(kp_single[1])])
                kp_tuple[str(i)] = kp_single_int
                i += 1

            #  Calculating angles #########################################
            kp_angle = []
            if type(kp) is np.ndarray:
                for kp_pair in self.pair:
                    kp_angle.append(self._angle(kp_tuple[str(kp_pair[0])],
                                                kp_tuple[str(kp_pair[1])],
                                                kp_tuple[str(kp_pair[2])])
                                    )

            # print('kp_angle is', kp_angle)
            return kp_angle, kp

        else:
            # print('No keypoints detected')
            return None, None
