import yaml
import cv2
import numpy as np

f = open("annotations.yaml", "r")
annos = yaml.load(f, Loader=yaml.Loader)["images"]
f.close()

files = list(annos)
files.sort()

for f in files:
    img = cv2.imread(f)
    y = 0
    for a in annos[f]["annotations"]:
        if not a["in_image"]:
            cv2.putText(img, f"{a['type']} not in image", (0, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            y += 20
        elif a["type"] == "robot":
            x_start = int(a["vector"][0][0])
            x_stop = int(a["vector"][1][0])
            y_start = int(a["vector"][1][0])
            y_stop = int(a["vector"][1][1])
            contours = np.array([(x_start, y_start), (x_start,y_stop), (x_stop, y_start), (x_stop, y_stop)])
            cv2.drawContours(img, contours, -1, (255,0,0), 5)

    cv2.imshow("img", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()


#debug = cv2.drawContours(cv2.UMat(img), [box.astype(int)], -1, (255, 255, 255), 10)
#cv2.imshow(key, debug)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
