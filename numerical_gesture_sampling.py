# This file is for recording dataset used for static gesture recognition

import cv2
import os

# Checking Dataset ####################################################
root_path = 'mazhiyuan_dataset'
class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
class_image_num = {class_name[0]: 0,
                   class_name[1]: 0,
                   class_name[2]: 0,
                   class_name[3]: 0,
                   class_name[4]: 0,
                   class_name[5]: 0,
                   class_name[6]: 0,
                   class_name[7]: 0,
                   class_name[8]: 0,
                   class_name[9]: 0,
                   }
if not os.path.exists(root_path):
    os.mkdir(root_path)
    print(root_path, 'is created')
else:
    print(root_path, 'exists')

for current_class_name in class_name:
    current_class_path = os.path.join(root_path, current_class_name)
    if not os.path.exists(current_class_path):
        os.mkdir(current_class_path)
        print(current_class_path, 'is created')
    else:
        print(current_class_path, 'exists')

        # Finding max val in image names ###############1#############
        class_image_num_list = os.listdir(current_class_path)
        max_num = 0
        if class_image_num_list:
            for current_class_image_num in class_image_num_list:
                tmp_num = int(current_class_image_num.split('.')[0])
                if max_num <= tmp_num:
                    max_num = tmp_num

            class_image_num[current_class_name] = max_num + 1
            print(current_class_name, 'has the image index up to', str(max_num))
        ############################################################
        else:
            print('This class is empty!')
####################################################################


cap = cv2.VideoCapture(0)  # 计算机自带的摄像头为0，外部设备为1
while True:
    ret, frame = cap.read()
    cv2.imshow("capture", frame)

    # Saving images ###########################################################

    # os.path.join(root_path, class_name[0])
    # cv2.imwrite('./test_set/%d.jpg' % class_image_num[0], frame)
    # print(class_image_num[0], '%d image(s) saved')
    # class_image_num[0] += 1

    for i in class_name:
        if (cv2.waitKey(1) & 0xFF) == ord(i):
            cv2.imwrite(os.path.join(root_path, class_name[int(i)], str(class_image_num[i]) + '.jpg'), frame)
            print(class_image_num[i], 'image(s) saved in', class_name[int(i)])
            class_image_num[i] += 1
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break
    ############################################################################

cap.release()
cv2.destroyAllWindows()
