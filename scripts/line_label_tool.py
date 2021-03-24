#!/usr/bin/env python3
import os
import numpy as np
import cv2

path="/home/florian/Projekt/bitbots/YOEO/data/Superset 1"
titles = ['No Green', 'Adaptive Gaussian Thresholding']
img = None

cv2.namedWindow('Adaptive Gaussian Thresholding')

cv2.createTrackbar('Weight1','Adaptive Gaussian Thresholding',10,150,lambda x: segment())
cv2.createTrackbar('ROI','Adaptive Gaussian Thresholding',40,100,lambda x: segment())
cv2.createTrackbar('Min Value','Adaptive Gaussian Thresholding',110,255,lambda x: segment())


def segment():
    global img

    tresh = cv2.getTrackbarPos('Weight1','Adaptive Gaussian Thresholding')
    roi = cv2.getTrackbarPos('ROI','Adaptive Gaussian Thresholding')
    min_val = cv2.getTrackbarPos('Min Value','Adaptive Gaussian Thresholding')

    normalized_roi = roi*10//2*2 + 1

    pad_img = np.pad(img[5:-5,5:-5], 100, mode="reflect")

    image_without_green = (0.5 * pad_img[..., 0] + 0.5 * pad_img[..., 2]).astype(np.uint8)

    blured_img = cv2.medianBlur(image_without_green, 3)
    #blured_img = cv2.medianBlur(blured_img,3)
    th3 = cv2.adaptiveThreshold(blured_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,normalized_roi,-tresh) - (255 - cv2.threshold(blured_img,min_val,255,cv2.THRESH_BINARY)[1])
    th3 = cv2.medianBlur(th3, 3)
    images = [image_without_green, th3]
    for i in range(len(images)):
        cv2.imshow(titles[i], images[i][100:-100,100:-100])

u = 0

for root,d_names,f_names in os.walk(path):
    if "masks" in root:
        continue


    f_names = [f for f in f_names if f.endswith(".png") or f.endswith(".jpg")]

    for f in f_names:
        u+=1
        if (not u%100==0):
            continue
        
        img_path = os.path.join(root, f)

        img = cv2.imread(img_path)

        cv2.imshow('Original', img)

        segment()

        cv2.waitKey(0)

cv2.destroyAllWindows()
