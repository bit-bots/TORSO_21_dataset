#!/usr/bin/env python3
import os
import numpy as np
import cv2


seg_in_path = ""
seg_out_path = ""


for f_name in os.listdir(seg_in_path):
    img = cv2.imread(os.path.join(seg_in_path, f_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Set all values larger than 127 to 255
    # This fixes a problem where the background and line color values were correct,
    # but the field colors where off.
    # The problem was caused by anti-aliasing of the field pixels with the line color
    # and an old bug of the line label tool.
    img[img>127] = 255

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(os.path.join(seg_out_path, f_name), img)
