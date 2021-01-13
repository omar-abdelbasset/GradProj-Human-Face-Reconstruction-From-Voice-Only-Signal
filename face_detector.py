import cv2
import dlib
import numpy as np
def detect(frame):
    """
    
    THIS FUNCTION DETECTS THE FACE IN GREY IMAGE AND CROP IT THEN PREDICT ITS  FACE FEATURE
    
    PAPAMETER:
    GREY:  np array; THE GREY SCALE OF IMAGE
    FRAME: np array; ACTUAL IMAGE
    
    RETURNS:
    FACE_FEATURE: np.array; PREDICTED ARRAY OF SIZE (1,2048)
    FRAME_CROP: np array; CROPED IMAGE
    
    """
    cnn_face_detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')
    faces_cnn = cnn_face_detector(frame, 1)
    for face in faces_cnn:
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y
    frame = frame[y:y+h,x:x + w]
    print(frame.shape)
    frame =  cv2.resize(frame, (224,224))
    print(frame.shape)
    #frame = np.array(frame.resize((224,224)).reshape((1,224,224,3)))
    #frame = np.expand_dims(frame, axis=0)
    return frame