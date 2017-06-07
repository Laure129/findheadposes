#!/usr/bin/env python
import tensorflow as tf
import cv2
from djangoapp.head_pose_estimation import CnnHeadPoseEstimator
from djangoapp.haar_cascade import haarCascade

class Estimator:
    def __init__(self, image_path):
        sess = tf.Session() #Launch the graph in a session.
        self.my_cascade = haarCascade(r"djangoapp/xml/haarcascade_frontalface_alt.xml",
                                 r"djangoapp/xml/haarcascade_profileface.xml")
        self.my_head_pose_estimator = CnnHeadPoseEstimator(sess) #Head pose estimation object
        self.my_head_pose_estimator.load_pitch_variables(r"djangoapp/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf")
        self.my_head_pose_estimator.load_yaw_variables(r"djangoapp/tensorflow/head_pose/yaw/cnn_cccdd_30k")
        self.image_path = image_path

    def get_pose(self):
    # найти лицо??
        image = cv2.imread(self.image_path) #Read the image with OpenCV

        image0 = cv2.imread(self.image_path,0)

        # Calling the findFace method

        self.my_cascade.findFace(image0, runFrontal=True, runFrontalRotated=True,
                            runLeft=True, runRight=True,
                            frontalScaleFactor=1.2, rotatedFrontalScaleFactor=1.2,
                            leftScaleFactor=1.15, rightScaleFactor=1.15,
                            minSizeX=64, minSizeY=64,
                            rotationAngleCCW=30, rotationAngleCW=-30)

        # The coords of the face are saved in the class object
        face_x1 = self.my_cascade.face_x
        face_y1 = self.my_cascade.face_y
        face_x2 = self.my_cascade.face_x + self.my_cascade.face_w
        face_y2 = self.my_cascade.face_y + self.my_cascade.face_h

        # Print this when no face is detected
        if (self.my_cascade.face_type == 0):
            #print("No face detected!")
            return [-180, -180, -180]
       # print(face_x1,face_x2,face_y1,face_y2)
       # print((face_y1 + face_y2) // 2, (face_x1 + face_x2) // 2)
       # print(image0.shape)
        # Делаем фото квадратным и в тоже время приближаем к координатам найденного лица
       # print(image0.shape[0]//2 - (face_y1 + face_y2) // 2, image0.shape[1]//2 - (face_x1 + face_x2) // 2)

        d1 = image0.shape[0] - (face_y1 + face_y2)
        if d1 < 0:
            image = image[0-d1:image.shape[0], 0:image.shape[1]]  # Crop from x, y, w, h -> 100, 200, 300, 400
        else:
            image = image[0:image.shape[0] -d1, 0:image.shape[1]]
        d2 = image0.shape[1] - (face_x1 + face_x2)
        if d2 < 0:
            image = image[0:image.shape[0], 0 - d2:image.shape[1]]  # Crop from x, y, w, h -> 100, 200, 300, 400
        else:
            image = image[0:image.shape[0], 0:image.shape[1] - d2]

        w = (image.shape[0] - image.shape[1]) // 2

        if w <0:
            image = image[0:image.shape[0], 0-w:image.shape[1]+w] # Crop from x, y, w, h -> 100, 200, 300, 400
        else:
            image = image[0+w:image.shape[0]-w, 0:image.shape[1]]  # Crop from x, y, w, h -> 100, 200, 300, 400
       # print(image.shape)
        image = cv2.resize(image, (min(image.shape[:-1]),min(image.shape[:-1])))

        pitch = self.my_head_pose_estimator.return_pitch(image) #Evaluate the pitch angle using a CNN
        yaw = self.my_head_pose_estimator.return_yaw(image) #Evaluate the yaw angle using a CNN
        return (None, pitch[0,0,0], yaw[0,0,0])

#print(Estimator(r'http://ftape.com/media/wp-content/uploads/2014/10/Sergei-Polunin_Numero-Homme_07.jpg').get_pose())