# This file is for evaluating the self-made dataset
# Make sure the dataset in `root_path` exists before running.

import cv2
import os
from hand_tracker import HandTracker
import numpy as np
import math


def root_check(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print(path, 'is created')
    else:
        print(root_path, 'exists')


class Evaluator:
    # Predict on images given
    def __init__(self):
        # Initializing the HandTracker #################################################
        palm_model_path = "./models/palm_detection.tflite"
        landmark_model_path = "./models/hand_landmark.tflite"
        anchors_path = "./data/anchors.csv"
        self.detector = HandTracker(palm_model_path, landmark_model_path, anchors_path,
                                    box_shift=0.1, box_enlarge=1.5)
        self.kp_single_int = tuple([0, 0])  # initialized as the result

        # Setting keypoints pairs ###############################
        self.pair = [[1, 2], [2, 3], [3, 4],
                     [5, 6], [6, 7], [7, 8],
                     [9, 10], [10, 11], [11, 12],
                     [13, 14], [14, 15], [15, 16],
                     [17, 18], [18, 19], [19, 20]]
        self.stem_pair = [0, 9]  # palm points and root middle-finger point

    @staticmethod
    def _angle(Y, X):
        # given cords of y and x in tuple format, returns arctan y x
        y1 = Y[1]
        y2 = X[1]
        x1 = Y[0]
        x2 = X[0]
        arctan = math.atan2(y1 - y2, x1 - x2)
        return arctan

    def predict(self, input_root, output_root):

        images_to_eval = os.listdir(input_root)
        for image in images_to_eval:
            image_num = image.split('.')[0]
            image = os.path.join(input_root, image)
            image = cv2.imread(image)

            # Calibrating Channels ################################
            b, g, r = cv2.split(image)
            image_cali = cv2.merge([r, g, b])
            #### Or using image_calis = image[:,:,::-1]

            # Predicting results #######################################
            kp, box = self.detector(image_cali)

            # Drawing output with dots ################################
            if type(kp) is np.ndarray:
                kp_tuple = {}
                kp_angle = []
                i = 0
                for kp_single in kp:
                    kp_single_int = tuple([int(kp_single[0]), int(kp_single[1])])
                    image = cv2.circle(image, kp_single_int, 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

                    # Meanwhile preparing for drawing lines ##################################
                    kp_tuple[str(i)] = kp_single_int
                    i += 1

                # Drawing output with lines ################################
                kp_angle = []
                if type(kp) is np.ndarray:
                    for kp_pair in self.pair:
                        image = cv2.line(image, kp_tuple[str(kp_pair[0])], kp_tuple[str(kp_pair[1])], (0, 255, 0),
                                         thickness=5)

                        # Meanwhile calculating angles #########################################
                        kp_angle.append(self._angle(kp_tuple[str(kp_pair[1])], kp_tuple[str(kp_pair[0])])
                                        - self._angle(kp_tuple[str(self.stem_pair[1])], kp_tuple[str(self.stem_pair[0])]))

                # Saving images with detected dots ##############################
                cv2.imwrite(os.path.join(output_root, image_num + '_eval.jpg'), image)
                print(os.path.join(output_root, image_num + '_eval.jpg'), 'saved. Key points are detected')
                print('kp_angle is', kp_angle)

            else:
                # Saving images with no detected dots ##############################
                cv2.imshow(os.path.join(output_root, image_num + '_eval.jpg'), image)
                print(os.path.join(output_root, image_num + '_eval.jpg'), 'saved. NO Key points')


if __name__ == '__main__':

    root_path = 'mazhiyuan_dataset'
    class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    class_path = {}

    root_check(root_path)
    for current_class_name in class_name:
        class_path[current_class_name] = os.path.join(root_path, current_class_name)

    for class_num_now in range(0, 10):
        # Setting a class as image input/ Creating output path ##########################
        image_set_from = class_path[str(class_num_now)]  # root path where input images gained
        image_output_to = os.path.join(root_path + '_eval',
                                       str(class_num_now) + '_eval')  # root path where results're output
        root_check(image_output_to)

        # Evaluating on the class ######################################################
        evaluator = Evaluator()
        evaluator.predict(image_set_from, image_output_to)
