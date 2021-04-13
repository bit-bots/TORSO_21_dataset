import yaml
import cv2
import numpy as np
import os
import pickle
import sys

MAIN_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ANNOTATION_INPUT_FILE = os.path.join(MAIN_FOLDER, 'data/annotations_with_metadata.yaml')
ANNOTATION_INPUT_FILE_PICKLED = os.path.join(MAIN_FOLDER, 'data/annotations_with_metadata.pkl')
with open(ANNOTATION_INPUT_FILE_PICKLED, "rb") as f:
    annos = pickle.load(f)["images"]
    print("Press 'a' and 's' to move between images. 'A' and 'S' let you jump 100 images.\n'c' to correct a label\n'v' to save image.\n'q' closes.\n'n' toggles not in image text. 'o' to toggle showing obstacles\n'e' to toggle all annotations")
    files = list(annos)
    files.sort()
    not_in_image = True
    show_obstacles = True
    show_annotations = True
    i = 0
    while True:
        f = files[i]
        img = cv2.imread(f)
        h, w, c = img.shape
        text_thickness = int(w/200)
        line_thickness = int(w/200)
        y = 20
        image_annos = annos[f]["annotations"]
        # sort lables to have them in the correct order. 
        image_annos_sorted = []
        correct_order = {"field edge": 0, "goalpost": 1, "left_goalpost": 2, "right_goalpost": 3, "top_bar": 4, "robot": 5, "obstacle": 6, "ball": 7, "L-Intersection": 8, "T-Intersection": 9, "X-Intersection": 10}
        for a in image_annos:
            a["order"] = correct_order[a["type"]]
        image_annos_sorted = sorted(image_annos, key=lambda a: a["order"])
        for a in image_annos_sorted:
            if show_annotations:
                if not a["in_image"]:
                    if not_in_image:
                        cv2.putText(img, f"{a['type']} not in image", (0, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), int(text_thickness/2))
                        y += 20
                else:
                    if a["type"] == "robot":
                        color = (255,0,0)
                    elif a["type"] == "ball":
                        color = (0,0,255)
                    elif a["type"] == "goalpost":
                        color = (0,255,255)
                    elif a["type"] == "left_goalpost":
                        color = (255,0,255)
                    elif a["type"] == "right_goalpost":
                        color = (0,255,255)
                    elif a["type"] == "top_bar":
                        color = (0,0,255)
                    elif a["type"] == "field edge":
                        color = (0,255,0)
                    elif a["type"] == "obstacle":
                        color = (0,0,0)

                    if a["type"] == "obstacle" and not show_obstacles:
                        pass
                    elif a["type"] == "robot" or a["type"] == "ball" or a["type"] == "obstacle": # bounding boxes
                        x_start = int(a["vector"][0][0])
                        x_stop = int(a["vector"][1][0])
                        y_start = int(a["vector"][0][1])
                        y_stop = int(a["vector"][1][1])
                        contours = np.ndarray((4,2), dtype=int)
                        contours[0][0] = x_start
                        contours[0][1] = y_start
                        contours[1][0] = x_start
                        contours[1][1] = y_stop
                        contours[2][0] = x_stop
                        contours[2][1] = y_stop
                        contours[3][0] = x_stop
                        contours[3][1] = y_start
                        cv2.drawContours(img, [contours], -1, color, line_thickness)
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
                        cv2.drawContours(img, [contours], -1, color, line_thickness)
                    elif a["type"] == "field edge":
                        points = []
                        for point in a["vector"]:
                            points.append(point)
                        pts = np.array(points, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        img = cv2.polylines(img, [pts], False, color, line_thickness)
                    else:
                        color = (0,0,0)
                        if a["type"] == "L-Intersection":
                            txt = "L"
                        elif a["type"] == "T-Intersection":
                            txt = "T"
                        elif a["type"] == "X-Intersection":
                            txt = "X"
                        else:
                            print(a["type"])
                            exit(1)
                        txt_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_COMPLEX, 1, text_thickness)
                        cv2.putText(img, txt, (int(a["vector"][0][0]-(txt_size[0][0]/2)), int(a["vector"][0][1]+(txt_size[0][1]/2))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), text_thickness)


        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        if key in [100] : #d
            i += 1
        elif key == 68: #D
            i += 100
        elif key in [115]: #s
            i -= 1
        elif key == 83: #S
            i -= 100
        elif key in [27, 113]:
            exit(0)
        elif key == 110: #n
            not_in_image = not not_in_image
        elif key == 111: #o
            show_obstacles = not show_obstacles
        elif key == 118: #v
            cv2.imwrite(f"../viz_{f}",img)
        elif key == 99: #c
            img_id = annos[f]['id']
            os.system(f"firefox https://imagetagger.bit-bots.de/annotations/{img_id}/ &")
        elif key == 101:
            show_annotations = not show_annotations
        i = max(0, i)
        i = min(len(files), i)
        sys.stdout.write("\x1b[A")
        sys.stdout.write("\x1b[A")
        print(f"Current image number {i} name {f}\n")


    cv2.destroyAllWindows()
