# The file for training and validation
from keras import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout, BatchNormalization
from keras.models import load_model, model_from_json
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import pandas as pd
import numpy as np
import os

data_set_path = 'mazhiyuan_dataset_angles.csv'
model_weight_name = 'model.hdf5'
model_structure_name = 'model.json'

# Setting params of optimizer
adam_lr = 0.0002
adam_beta_1 = 0.5

def rescale(angle_data_x):
    # Scaling the data ######################################
    data_scaled = np.zeros(shape=(angle_data_x.shape[0], angle_data_x.shape[1]))
    for i in range(angle_data_x.shape[1]):
        data_toscale = angle_data_x[:, i]
        mean = np.mean(data_toscale)
        var = np.var(data_toscale)
        data_toadd = (data_toscale - mean) * (data_toscale - mean) / var
        data_toadd = np.expand_dims(data_toadd, 0)
        data_scaled[:, i] = data_toadd
        # print('hold on')

    return data_scaled

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class CSV_creator:
    def __init__(self):
        self.path = data_set_path
        self.heading = ['thumb_low', 'thumb_middle', 'thumb_high',
                        'index_low', 'index_middle', 'index_high',
                        'middle_low', 'middle_middle', 'middle_high',
                        'ring_low', 'ring_middle', 'ringle_high',
                        'pinky_low', 'pinky_middle', 'pinky_high']

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


class Mediapipe_Detector():
    def __init__(self, X_train, Y_train, X_test, Y_test):
        # Importing package for training and running models
        self.path = data_set_path
        self.classifier = None
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def label_encoder(self):
        encoder = LabelEncoder()
        encoder.fit(self.Y_train)
        Y_train_encoded = encoder.transform(self.Y_train)
        self.Y_train = np_utils.to_categorical(Y_train_encoded)

        encoder.fit(self.Y_test)
        Y_test_encoded = encoder.transform(self.Y_test)
        self.Y_test = np_utils.to_categorical(Y_test_encoded)

    def train(self):
        # if os.path.exists(model_weight_name) and os.path.exists(model_structure_name):
        #     # Loading the model #######################################
        #     json_file = open(model_structure_name, 'r')
        #     classifier_json = json_file.read()
        #     json_file.close()
        #     self.classifier = model_from_json(classifier_json)
        #     self.classifier.load_weights(model_weight_name)

        # else:
        #     # # Building the dense layer model #######################################
        #     # self.classifier = Sequential()
        #     # self.classifier.add(Dense(20, activation='relu',
        #     #                           kernel_initializer='random_normal',
        #     #                           input_dim=10))
        #     # # self.classifier.add(Conv1D(100, 2, 2, activation='relu',
        #     # #                            kernel_initializer='relu'))
        #     # self.classifier.add(Dense(12, activation='relu',
        #     #                           kernel_initializer='random_normal'))
        #     # self.classifier.add(Dense(10, activation='relu',
        #     #                           kernel_initializer='random_normal'))

        # Building the CNN layer model ########################
        self.classifier = Sequential()
        self.classifier.add(Conv1D(1000,
                                   2,
                                   strides=2,
                                   padding='valid',
                                   activation='relu',
                                   use_bias=True,
                                   kernel_initializer='random_normal',
                                   input_shape=(15, 1),
                                   name='convolution_1d_layer_1'))
        self.classifier.add(Conv1D(1000,
                                   5,
                                   padding='valid',
                                   activation='relu',
                                   use_bias=True,
                                   kernel_initializer='random_normal',
                                   name='convolution_1d_layer_2'))

        self.classifier.add(Flatten(name='reshape_layer'))
        self.classifier.add(Dense(100,
                                  kernel_initializer='random_normal',
                                  # bias_initializer=' random_normal',
                                  use_bias=True,
                                  name='full_connect_layer_1',
                                  activation='softmax'))
        # self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(10,
                                  kernel_initializer='random_normal',
                                  # bias_initializer=' random_normal',
                                  use_bias=True,
                                  name='full_connect_layer_2',
                                  activation='softmax'))


        self.classifier.compile(optimizer= Adam(adam_lr),
                                # loss='binary_crossentropy',
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])

        # Record acc on the log #################################
        losshistory = LossHistory()

        # Saving the model if the validation loss decresed
        checkpointer = ModelCheckpoint(filepath=model_weight_name, verbose=1, save_best_only=True)

        # Fitting/training the model ##############################
        self.label_encoder()
        print(self.Y_train)  # just for test
        self.classifier.fit(self.X_train, self.Y_train, epochs=2000, validation_data=(self.X_test, self.Y_test),
                            callbacks=[checkpointer, losshistory], batch_size=80)

        # Reporting loss history ##################################
        print(losshistory.losses)

        # Serializing model to JSON ##############################
        classifier_json = self.classifier.to_json()
        with open(model_structure_name, 'w') as json_file:
            json_file.write(classifier_json)


if __name__ == '__main__':
    # Loading angle data from .csv file #############################
    csv_creator = CSV_creator()
    angle_data_x, angle_data_y = csv_creator.load()

    # Scaling input data ################
    angle_data_x = rescale(angle_data_x)



    # # Reformatting angle_data_x under the constrains of Conv1D ####
    tmp_angle_data_x = angle_data_x
    angle_data_x = np.empty(shape=[0, 15, 1], dtype=float)
    for singe_line in tmp_angle_data_x:
        singe_line = np.expand_dims(singe_line, 0)
        singe_line = np.expand_dims(singe_line, 2)
        angle_data_x = np.append(angle_data_x, singe_line, axis=0)

    # Splitting the dataset ####################################
    x_train, x_test, y_train, y_test = train_test_split(angle_data_x, angle_data_y, test_size=0.1, random_state=0)

    # Training the model ############################
    detector = Mediapipe_Detector(x_train, y_train, x_test, y_test)
    detector.train()
    # print('hold on')
