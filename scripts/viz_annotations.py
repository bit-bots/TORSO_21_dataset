import yaml
import cv2
import numpy as np
import os

MAIN_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ANNOTATION_INPUT_FILE = os.path.join(MAIN_FOLDER, 'data/annotations_with_metadata.yaml')
with open(ANNOTATION_INPUT_FILE) as f:
    annos = yaml.load(f, Loader=yaml.Loader)["images"]

    files = list(annos)
    files.sort()

    for f in files:
        img = cv2.imread(f)
        y = 0
        for a in annos[f]["annotations"]:
            if not a["in_image"]:
                cv2.putText(img, f"{a['type']} not in image", (0, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                y += 20
            else:
                if a["type"] == "robot":
                    color = (255,0,0)
                elif a["type"] == "ball":
                    color = (0,255,0)
                elif a["type"] == "left_goalpost":
                    color = (255,0,255)
                elif a["type"] == "right_goalpost":
                    color = (0,255,255)
                elif a["type"] == "top_bar":
                    color = (0,0,255)

                if a["type"] == "robot" or a["type"] == "ball": # bounding boxes
                    x_start = int(a["vector"][0][0])
                    x_stop = int(a["vector"][1][0])
                    y_start = int(a["vector"][0][1])
                    y_stop = int(a["vector"][1][1])
                    print(f"x_start: {x_start}, x_stop: {x_stop}, y_start: {y_start}, y_stop: {y_stop}")
                    contours = np.ndarray((4,2), dtype=int)
                    contours[0][0] = x_start
                    contours[0][1] = y_start
                    contours[1][0] = x_start
                    contours[1][1] = y_stop
                    contours[2][0] = x_stop
                    contours[2][1] = y_stop
                    contours[3][0] = x_stop
                    contours[3][1] = y_start
                elif a["type"] == "goalpost" or a["type"] == "left_goalpost" or a["type"] == "right_goalpost" or a["type"] == "top_bar":
                    contours = np.ndarray((4, 2), dtype=int)
                    contours[0][0] = int(a["vector"][0][0])
                    contours[0][1] = int(a["vector"][0][1])
                    contours[1][0] = int(a["vector"][1][0])
                    contours[1][1] = int(a["vector"][1][1])
                    contours[2][0] = int(a["vector"][2][0])
                    contours[2][1] = int(a["vector"][2][1])
                    contours[3][0] = int(a["vector"][3][0])
                    contours[3][1] = int(a["vector"][3][1])
                else:
                    continue#todo intersections not handled

                cv2.drawContours(img, [contours], -1, color, 5)


        cv2.imshow("img", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
