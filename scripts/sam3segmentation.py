import torch
import os
from os import listdir
#################################### For Image ####################################
from PIL import Image
import PIL.Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import numpy as np 
import cv2
import torch
import matplotlib.pyplot as plt
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
folder_dir = "/homes/25kelmend/testPic/"

folder = "path/to/images"



for image_path in os.listdir(folder_dir):
    if "mask" in image_path.lower():
        continue  # skip mask images
    rgba_image = PIL.Image.open("/homes/25kelmend/testPic/" + image_path)
    rgb_image = rgba_image.convert('RGB')
    #image = Image.open(rgb_image)
    inference_state = processor.set_image(rgb_image)

        # Prompt the model with text
    output = processor.set_text_prompt(state=inference_state, prompt="line marks, particularly side lines, also notice fine lines far away, intersections, white markers")

        # Get the masks, bounding boxes, and scores
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    print(masks.shape)

    if masks.shape[0] == 0:
        continue

    mask_np = masks.detach().cpu().numpy()

    mask_np = np.any(mask_np, axis=0).squeeze(0)


    plt.imshow(mask_np, cmap='gray')
    plt.title("Mask")
    plt.show()
    plt.imsave(folder_dir + image_path + "_mask.png", mask_np, cmap='gray')