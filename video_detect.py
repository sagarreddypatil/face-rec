import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from math import *
from PIL import Image as Image
import PIL.ImageOps
import os
from model import Model

names = ["Sagar", "Other"]
model = Model()
model.load(path_to_model = "model/model.h5")

print("Model Loaded")

def midpoint(p1, p2):
    x_mid = (p1[0]+p2[0])/2
    y_mid = (p1[1]+p2[1])/2
    
    return (x_mid, y_mid)

def rotate_helper(x, y, angle):
    rad = radians(angle)
    x_new = cos(rad) * x - sin(rad) * y
    y_new = sin(rad) * x + cos(rad) * y
    return x_new, y_new

def rotate(point, angle, center):
    moved_x = point[0] - center[0]
    moved_y = point[1] - center[1]

    rotated_x, rotated_y = rotate_helper(moved_x, moved_y, angle)

    back_x = rotated_x + center[0]
    back_y = rotated_y + center[1]

    return (int(back_x), int(back_y))

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture('ya.mov')

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output.avi', -1, 20.0, (1280,720))


while(cap.isOpened()):
    ret, img = cap.read()
    if ret==True:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 10)

        for (x,y,w,h) in faces:
            face_color = img[y:y+h, x:x+h]
            face_gray = gray[y:y+h, x:x+h]

            eyes = eye_cascade.detectMultiScale(face_gray, 5, 10)

            first_eye_center = (0, 0)
            second_eye_center = (0, 0)

            i = 0

            for (ex, ey, ew, eh) in eyes:
                #cv2.rectangle(face_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
                mid_x = ceil((ex+(ex+ew))/2)
                mid_y = ceil((ey+(ey+eh))/2)

                if i == 0:
                    first_eye_center = (mid_x,mid_y)
                else:
                    second_eye_center = (mid_x, mid_y)
                i = i + 1
            
            deg = 0
            if first_eye_center == (0, 0) or second_eye_center == (0, 0) or first_eye_center == second_eye_center:
                pass
            else:
                try:
                    cv2.circle(face_color, first_eye_center, 1, (0, 255, 0), thickness=10, lineType=8, shift=0)
                    cv2.circle(face_color, second_eye_center, 1, (0, 255, 0), thickness=10, lineType=8, shift=0)
                    cv2.line(face_color, first_eye_center, second_eye_center, (255, 0, 0), 1)
                    deg = int(atan((first_eye_center[1] - second_eye_center[1]) / (first_eye_center[0] - second_eye_center[0]))*180/pi)
                    if deg >= 27 or deg <= -27:
                        deg = 0
                    #print(deg, end="\r")
                except:
                    pass
            top_left = (x, y)
            bottom_left = (x, y+h)
            top_right = (x+w, y)
            bottom_right = (x+w, y+h)
            center = midpoint(top_left, bottom_right)

            top_left_new = rotate(top_left, deg, center)
            bottom_left_new = rotate(bottom_left, deg, center)
            top_right_new = rotate(top_right, deg, center)
            bottom_right_new = rotate(bottom_right, deg, center)

            cv2.line(img, top_left_new, top_right_new, (255, 0, 0), 1)
            cv2.line(img, top_right_new, bottom_right_new, (255, 0, 0), 1)
            cv2.line(img, bottom_right_new, bottom_left_new, (255, 0, 0), 1)
            cv2.line(img, bottom_left_new, top_left_new, (255, 0, 0), 1)

            rows, cols = img.shape[0], img.shape[1]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), deg, 1)
            img_rotated = cv2.warpAffine(img, M, (cols, rows))
            #cv2.rectangle(img_rotated, top_left, bottom_right, (0, 0, 255))
            final_face = img_rotated[y:y+h, x:x+w]
            final_gray = cv2.cvtColor(final_face, cv2.COLOR_BGR2GRAY)
            final_gray = cv2.resize(final_gray, (128, 128), interpolation = Image.ANTIALIAS)
            final_face = cv2.resize(final_face, (128, 128), interpolation = Image.ANTIALIAS)
            X = np.array([final_face])
            pred_prob = model.predict(X)
            pred_id = np.argmax(pred_prob, axis=1)[0]
            pred_label = names[pred_id]

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, pred_label + " " + str(pred_prob[0][pred_id]), top_right_new, font, 1, (255, 0, 0), 1, cv2.LINE_AA)

            out.write(img)
            cv2.imshow('face', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()