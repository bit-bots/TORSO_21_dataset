#!/usr/bin/env python3
import os
import time
import numpy as np
import cv2


class LineLabelTool(object):
    def __init__(self):
        self.path="/home/florian/Projekt/bitbots/YOEO/data/Superset 1"
        self.titles = ['No Green', 'Adaptive Gaussian Thresholding']

        self.img = None
        self.segmentation = None
        self._box_size = 20
        self._mouse_coord = (0,0)
        self._left_click = False

        cv2.namedWindow('Adaptive Gaussian Thresholding')

        cv2.createTrackbar('Weight1','Adaptive Gaussian Thresholding',10,150,lambda x: self.segment())
        cv2.createTrackbar('ROI','Adaptive Gaussian Thresholding',40,100,lambda x: self.segment())
        cv2.createTrackbar('Min Value','Adaptive Gaussian Thresholding',110,255,lambda x: self.segment())

        cv2.namedWindow('Segmentation')

        cv2.setMouseCallback("Segmentation", self._mouse_callback)


    def segment(self):
        tresh = cv2.getTrackbarPos('Weight1','Adaptive Gaussian Thresholding')
        roi = cv2.getTrackbarPos('ROI','Adaptive Gaussian Thresholding')
        min_val = cv2.getTrackbarPos('Min Value','Adaptive Gaussian Thresholding')

        normalized_roi = roi*10//2*2 + 1

        pad_img = np.pad(self.img[5:-5,5:-5], 100, mode="reflect")

        image_without_green = (0.5 * pad_img[..., 0] + 0.5 * pad_img[..., 2]).astype(np.uint8)

        blured_img = cv2.medianBlur(image_without_green, 3)
        #blured_img = cv2.medianBlur(blured_img,3)
        segmentation = cv2.adaptiveThreshold(blured_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,normalized_roi,-tresh) - (255 - cv2.threshold(blured_img,min_val,255,cv2.THRESH_BINARY)[1])
        self.segmentation = cv2.medianBlur(segmentation[100:-100,100:-100], 3)
        images = [image_without_green[100:-100,100:-100], self.segmentation]
        for i in range(len(images)):
            cv2.imshow(self.titles[i], images[i])

    def edit(self):
        history = [self.segmentation.copy(),]
        while True:
            # Copy image for the canvas
            canvas = history[-1].copy()
            # Get the mouse coordinates
            x, y = self._mouse_coord

            # Get the values for the selection box
            box_min_x = max(int(x - self._box_size / 2), 0)
            box_min_y = max(int(y - self._box_size / 2), 0)
            box_max_x = min(int(x + self._box_size / 2), self.segmentation.shape[1])
            box_max_y = min(int(y + self._box_size / 2), self.segmentation.shape[0])

            # Check for click event
            if self._left_click:
                # Delete pixels in selection
                canvas[
                    box_min_y : box_max_y,
                    box_min_x : box_max_x
                ] = 0
                
                # Append new color space to self._history
                self.segmentation = canvas.copy()

                history.append(self.segmentation)

                # Reset events
                self._left_click = False

            # Draw selection area
            cv2.rectangle(canvas,
                (box_min_x, box_min_y),
                (box_max_x, box_max_y),
                (255, 255, 255),
                2)

            # Show canvas
            cv2.imshow("Segmentation", canvas)

            # Key checks for UI events
            key = cv2.waitKey(1) & 0xFF

            # Increase selection box size
            if key == ord("+"):
                self._box_size += int(self._box_size * 0.2) + 1
            # Reduce selection box size
            if key == ord("-") and self._box_size > 1:
                self._box_size -= int(self._box_size * 0.2) - 1
            # Undo
            if key == ord("u") and len(history) > 1:
                del history[-1]
            # Quit exit
            elif key == ord('s'):
                break

            time.sleep(0.01)


    def _mouse_callback(self, event, x, y, flags, param):
            """
            Callback for the OpenCV cursor.

            :param event: Event type
            :param x: Mouse x position
            :param y: Mouse y position
            :param flags: Callback flags
            :param param: Some unused parameter
            """
            self._left_click = (event == cv2.EVENT_LBUTTONUP)

            # Set self._mouse_coordinates
            self._mouse_coord = (x, y)


    def main_loop(self):
        u = 0

        for root,_,f_names in os.walk(self.path):
            if "masks" in root:
                continue


            f_names = [f for f in f_names if f.endswith(".png") or f.endswith(".jpg")]

            for f in f_names:
                u+=1
                if (not u%100==0):
                    #continue
                    pass
                
                img_path = os.path.join(root, f)

                self.img = cv2.imread(img_path)

                cv2.imshow('Original', self.img)

                self.segment()

                cv2.waitKey(0)

                self.edit()

                cv2.imwrite(os.path.join('/tmp/', os.path.join(os.path.splitext(os.path.basename(f))[0] + '.png')), self.segmentation) 

        cv2.destroyAllWindows()


if __name__ == "__main__":
    LineLabelTool().main_loop()

