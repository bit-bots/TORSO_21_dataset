import os
import cv2
import numpy as np
import yaml
import pickle
from collections import defaultdict
from tqdm import tqdm


class YoloEvalOpenCV():
    """
    Opencv library implementation of our yolo model.
    """
    def __init__(self, model_path, image_folder, annotation_file):
        """
        Initialization of the YoloEvalOpenCV

        :param config:
        :param model_path:
        """
        self._confidence_threshold = 0.25
        self._nms_threshold = 0.5

        with open(annotation_file, 'rb') as f:
            self.annotations = yaml.load(f)

        self.image_folder = image_folder

        # Build paths
        weightpath = os.path.join(model_path, "yolo_weights.weights")
        configpath = os.path.join(model_path, "config.cfg")
        self._classes = ["ball", "goalpost", "robot", "L-Intersection", "T-Intersection", "X-Intersection"]
        # Setup neural network
        self._net = cv2.dnn.readNet(weightpath, configpath)

    def _get_output_layers(self):
        """
        Library stuff
        """
        layer_names = self._net.getLayerNames()

        output_layers = [layer_names[i[0] - 1] for i in self._net.getUnconnectedOutLayers()]

        return output_layers

    def predict(self, image):
        """
        Runs the neural network
        """
        # Set image
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self._net.setInput(blob)
        self._width = image.shape[1]
        self._height = image.shape[0]
        # Run net
        self._outs = self._net.forward(self._get_output_layers())
        # Create lists
        class_ids = []
        confidences = []
        boxes = []
        # Iterate over output/detections
        for out in self._outs:
            for detection in out:
                # Get score
                scores = detection[5:]
                # Ger class
                class_id = np.argmax(scores)
                # Get confidence from score
                confidence = scores[class_id]
                # First threshold to decrease candidate count and inscrease performance
                if confidence > self._confidence_threshold:
                    # Get center point of the candidate
                    center_x = int(detection[0] * self._width)
                    center_y = int(detection[1] * self._height)
                    # Get the heigh/width
                    w = int(detection[2] * self._width)
                    h = int(detection[3] * self._height)
                    # Calc the upper left point
                    x = center_x - w / 2
                    y = center_y - h / 2
                    # Append result
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Merge boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self._confidence_threshold, self._nms_threshold)

        prediction_segmentation = np.zeros((len(self._classes), self._height, self._width))

        # Iterate over filtered boxes
        for i in indices:
            # Get id
            i = i[0]
            # Get box
            box = boxes[i]
            # Convert the box position/size to int
            box = list(map(int, box))
        
            prediction_segmentation[class_ids[i]] = cv2.rectangle(
                prediction_segmentation[class_ids[i]],
                (box[0], box[1]),
                (box[0] + box[2], box[1]+ box[3]), 255, -1)

        return prediction_segmentation

    
    def generate_annotation_segmentation(self, annotations, shape):

        annotation_segmentation = np.zeros((shape), dtype=np.uint8)

        for class_idx, annotation_type in enumerate(self._classes):

            annotations_of_type_in_image = filter(
                lambda annotation: 
                    annotation_type == annotation['type'] and annotation['in_image'],
                annotations)

            for annotation in annotations_of_type_in_image:

                vector = np.array(annotation['vector'], dtype = np.int32)

                # Bbox
                if annotation_type in  ['robot', 'ball']:
                    annotation_segmentation[class_idx] = cv2.rectangle(
                        annotation_segmentation[class_idx], tuple(vector[1].tolist()), tuple(vector[0].tolist()), 255, -1)
                # Polygon
                elif annotation_type == 'goalpost':
                    annotation_segmentation[class_idx] = cv2.fillConvexPoly(annotation_segmentation[class_idx], vector, 255.0)
                # Keypoint
                elif 'Intersection' in annotation_type:
                    size_h = int(self._height * 0.03)
                    size_w = int(self._width * 0.03)
                    annotation_segmentation[class_idx] = cv2.rectangle(
                        annotation_segmentation[class_idx], 
                        (vector[0][0]-size_w, vector[0][1]-size_h),
                        (vector[0][0]+size_w, vector[0][1]+size_h), 255, -1)
        
        return annotation_segmentation

    
    def _match_masks(self, label_mask, detected_mask):
        """
        Calculates the iou
        """
        label_mask = label_mask.astype(bool)
        detected_mask = detected_mask.astype(bool)
        numerator = float(np.sum(np.bitwise_and(label_mask, detected_mask)))
        denominator = float(np.sum(np.bitwise_or(label_mask, detected_mask)))
        iou = numerator / denominator if denominator > 0 else None
        return iou


    def run(self):
        ious_for_class = defaultdict(list)
        for name, annotations in tqdm(self.annotations['images'].items()):

            annotations = annotations['annotations']

            image_path = os.path.join(self.image_folder, name)

            image = cv2.imread(image_path)

            if image is None:
                print("Image not found")
                continue

            cv2.imshow("Image",image)

            prediction_segmentation = self.predict(image)

            annotation_segmentation = self.generate_annotation_segmentation(annotations, prediction_segmentation.shape)

            for class_idx, cls_name in enumerate(self._classes):
                canvas = np.zeros((*prediction_segmentation[class_idx].shape, 3), dtype=np.uint8)
                canvas[:,:,0] = prediction_segmentation[class_idx]
                canvas[:,:,2] = annotation_segmentation[class_idx]
                cv2.imshow(f"{cls_name}_seg",cv2.resize(canvas, (600,400)))
                cv2.imshow(f"Image",cv2.resize(image, (600,400)))
                cv2.waitKey(1)
                iou = self._match_masks(annotation_segmentation[class_idx], prediction_segmentation[class_idx])
                if iou is not None:
                    ious_for_class[cls_name].append(iou)
                else:
                    ious_for_class[cls_name].append(1.0)

        miou_for_class = {class_name: (np.array(ious).mean(), np.array(ious).std()) for class_name, ious in ious_for_class.items()}

        print(miou_for_class)

            
if __name__ == "__main__":
    YoloEvalOpenCV(
        "/home/florian/Downloads/real_tiny_yolo", 
        "/home/florian/Downloads/imageset/", 
        "/home/florian/Downloads/vision_dataset_2021_test_labels.yaml").run()

    

