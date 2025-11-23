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
#folder_dir = "/homes/25kelmend/testPic/"
folder_dir = "/homes/25kelmend/testimg/"

folder = "path/to/images"
#        #line marks, particularly side lines, also notice fine lines far away, intersections, white markers


for image_path in os.listdir(folder_dir):
    if "mask" in image_path.lower():
        continue  # skip mask images
    rgba_image = PIL.Image.open(folder_dir + image_path)
    rgb_image = rgba_image.convert('RGB')
    #image = Image.open(rgb_image)
    inference_state = processor.set_image(rgb_image)
    prompts = ["white lines", "white intersecting lines on grass", "white line marks on grass", "far away white line marks on grass", "white lines on grass", "white lines on grass at the edge of the image"]


    output = processor.set_text_prompt(state=inference_state, prompt= prompts[0])
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    comb_mask = masks.detach().cpu().numpy()
    comb_mask = np.any(comb_mask, axis=0).squeeze(0)

    print(image_path)

    for line_prompt in prompts:
        output = processor.set_text_prompt(state=inference_state, prompt= line_prompt)
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        print(line_prompt + "   " + str(masks.shape[0]))
        mask_np = masks.detach().cpu().numpy()
        mask_np = np.any(mask_np, axis=0).squeeze(0)
        comb_mask = mask_np | comb_mask
        """
        plt.imshow(mask_np, cmap='gray')
        plt.title("Mask")
        plt.show()
        plt.imsave(folder_dir + image_path + line_prompt + "_mask.png", mask_np, cmap='gray')
        """
    print("")

    plt.imshow(comb_mask, cmap='gray')
    plt.title("Mask")
    plt.show()
    plt.imsave(folder_dir + image_path +  "_mask.png", comb_mask, cmap='gray')