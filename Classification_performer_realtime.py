import cv2
from keras.models import model_from_json
from kpdata_extractor_imageshow import Extractor_v2
import numpy as np

model_weight_name = 'model.hdf5'
model_structure_name = 'model.json'
show_gesture = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
                5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}

pair = [[0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16],
        [0, 17], [17, 18], [18, 19], [19, 20]]  # Skeleton based on key points pairs
font = cv2.FONT_HERSHEY_SIMPLEX

if __name__ == '__main__':

    # Loading models ################################
    json_file = open(model_structure_name, 'r')
    Loaded_json_file = json_file.read()
    json_file.close()
    classifier = model_from_json(Loaded_json_file)
    classifier.load_weights(model_weight_name)

    # Initializing angles detectors #####################
    extractor = Extractor_v2()

    # Initializing camera ##############################
    cap = cv2.VideoCapture(0)

    # Looping the detection #########################
    while True:
        retval, frame = cap.read()
        if retval:

            # Performing detection #######################
            angles, kp = extractor.run(frame)
            if angles:
                angles_np = np.array(angles)

                # Reformatting the input ###########
                angles_np = np.expand_dims(angles_np, 0)
                angles_np = np.expand_dims(angles_np, 2)

                # Starting the prediction ###########
                result = classifier.predict(angles_np)

                # Showing the result #######################
                max_index = np.argmax(result[0])

                # Drawing the prediction results on the image ####################
                cv2.putText(frame, show_gesture[ max_index], (40, 40), font, 1.5, (0, 255, 0), 3)
                cv2.putText(frame, str(result[0, max_index]), (200, 40), font, 1, (0, 255, 0), 3)

                # Drawing output with lines ################################
                kp_tuple = {}
                i = 0
                for kp_single in kp:
                    kp_single_int = tuple([int(kp_single[0]), int(kp_single[1])])
                    frame = cv2.circle(frame, kp_single_int, 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

                    # BTW preparing for drawing lines ##################################
                    kp_tuple[str(i)] = kp_single_int
                    i += 1


                for kp_pair in pair:
                    frame = cv2.line(frame, kp_tuple[str(kp_pair[0])], kp_tuple[str(kp_pair[1])], (0, 255, 0),
                                     thickness=5)

            cv2.imshow('video', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
