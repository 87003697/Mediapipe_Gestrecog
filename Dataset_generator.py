from kpdata_extractor_imageshow import Extractor_v2
import os
import csv
import cv2
import pandas as pd
import numpy as np


data_set_path = 'mazhiyuan_dataset_angles.csv'


def content_exist(reader):
    # Check if something exists in current .csv file
    for line in reader:
        if line:
            return True
        else:
            return False


class CSV_creator:

    def __init__(self):
        self.path = data_set_path
        self.csv_reader = None
        self.csv_writer = None
        self.extractor = Extractor_v2()
        self.heading = ['thumb_low', 'thumb_middle', 'thumb_high',
                        'index_low', 'index_middle', 'index_high',
                        'middle_low', 'middle_middle', 'middle_high',
                        'ring_low', 'ring_middle', 'ringle_high',
                        'pinky_low', 'pinky_middle', 'pinky_high']

    def reset(self):
        # Loading the heading/ Resetting the file ##############################
        heading = self.heading
        heading.append('class')
        with open(self.path, 'w', newline='') as file:
            self.csv_writer = csv.writer(file)
            self.csv_writer.writerow(heading)

    @staticmethod
    def listofstr2listoflist(list):
        # convert a list of strings to a list of list
        listoflist = []
        for single_line in list:
            single_line_list = int(single_line)
            listoflist.append(single_line_list)
        return listoflist

    def load(self):
        """
        Load all angle data from .csv file
        :return: angle_data: the angles data for recognition
        """

        angle_data = pd.read_csv(self.path)
        angle_data_values = angle_data.values
        angle_data_x = angle_data_values[:, 0:len(self.heading)]
        angle_data_y = angle_data_values[:, len(self.heading)]

        return angle_data_x, angle_data_y

    def create(self, path, class_name):
        """
        Extract angles out of raw images, and save the angles to the .csv file on the `path`
        :param path:
        :param class_name:
        :return:
        """
        images_namelist = os.listdir(path)

        for image_name in images_namelist:
            image = cv2.imread(os.path.join(path, image_name))

            # Calibrating Channels ################################
            b, g, r = cv2.split(image)
            image_cali = cv2.merge([r, g, b])

            # Extracting and Saving features #################################
            angles = self.extractor.run(image_cali)

            # single_csv_towrite = [angles, class_name]
            if angles:
                angles.append(class_name)
                single_csv_towrite = angles

            # Storing features in .csv file ###############################
            if angles:
                with open(self.path, 'a+', newline='') as file:
                    self.csv_writer = csv.writer(file, dialect='excel')
                    self.csv_writer.writerow(single_csv_towrite)


if __name__ == '__main__':
    root_path = 'mazhiyuan_dataset'
    class_name = os.listdir(root_path)
    csv_creator = CSV_creator()

    # Operating on the whole class #########################
    csv_creator.reset()
    for single_class_name in class_name:
        single_class_path = os.path.join(root_path, single_class_name)
        csv_creator.create(single_class_path, int(single_class_name))


