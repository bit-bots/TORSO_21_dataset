from matplotlib import patches
from matplotlib.colors import ListedColormap
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
#folder_dir = "/homes/25kelmend/Downloads/sam3_images/testimg_large_prompt2/"
#folder_dir = "/homes/25kelmend/Downloads/sam3_images/testimg_mainprompt_overlay/"
#folder_dir = "/homes/25kelmend/Downloads/sam3_images/testimg_large_robot/"
folder_dir = "/homes/25kelmend/Downloads/sam3_images/testimg_large_mainprompt/"
#folder_dir = "/homes/25kelmend/Downloads/sam3_images/testimg_prompt2/"

folder = "path/to/images"
#prompts:
#lineprompt: ["white lines","white intersecting lines on grass","white line marks on grass","far away white line marks on grass","white lines on grass","white lines on grass at the edge of the image"]  
#lineprompt2: ["white intersecting lines on grass","white line marks on grass","far away white line marks on grass","white lines on grass","white lines on grass at the edge of the image"]]
#promptRobot: ["robot on the field", "humanoid robot on grass", "soccer robot on grass", "robot on grass"]
#fieldprompt: ["football field", "field", "grass field"]

####################################image counter################################################

images_amount = 0
image_current = 0
for file in os.scandir(folder_dir):
    if "mask" in file.name.lower():
        continue
    else:
        images_amount +=1

print(f"Amount of images: {images_amount}")
print("************************")

####################################for loop start###############################################
for image_entry in os.scandir(folder_dir): #Frage: Wäre scandir besser? Hab gelesen bessere performance
    if "mask" in image_entry.name.lower():
        continue  # skip mask images
    if image_entry.is_file():
        rgba_image = PIL.Image.open(image_entry.path)
        rgb_image = rgba_image.convert('RGB')
        #image = Image.open(rgb_image)
        inference_state = processor.set_image(rgb_image)
        print(image_entry.path)

###################################Mask lines generation##########################################

    prompts_lines = ["white intersecting lines on grass",
               "white line marks on grass",
               "far away white line marks on grass",
               "white lines on grass",
               "white lines on grass at the edge of the image"]
    
    print("************************")
    print("White lines detection")
    print("************************")

    output_lines = processor.set_text_prompt(state=inference_state, prompt= prompts_lines[0])
    masks, boxes, scores = output_lines["masks"], output_lines["boxes"], output_lines["scores"]
    line_mask = masks.detach().cpu().numpy()
    line_mask = np.any(line_mask, axis=0).squeeze(0)
    #Frage: Brauch ich any(...) wenn es ja nur ein Prompt ist?. Kann ich schreiben
    #line_mask = masks.detach().cpu().numpy().squeeze(0)

    for line_prompt in prompts_lines:
        output_lines = processor.set_text_prompt(state=inference_state, prompt= line_prompt)
        masks, boxes, scores = output_lines["masks"], output_lines["boxes"], output_lines["scores"]
        print(line_prompt + "   " + str(masks.shape[0]))
        mask_np = masks.detach().cpu().numpy()
        mask_np = np.any(mask_np, axis=0).squeeze(0)
        line_mask = mask_np | line_mask

        """
        plt.imshow(mask_np, cmap='gray')
        plt.title("Mask")
        plt.show()
        plt.imsave(folder_dir + image_path + line_prompt + "_mask.png", mask_np, cmap='gray')
        """

    #########################################For Robots########################################
    prompts_robots = ["robot on the field", "humanoid robot on grass", "soccer robot on grass", "robot on grass"]
    print("************************")
    print("Robot detection")
    print("************************")

    output_robots = processor.set_text_prompt(state=inference_state, prompt= prompts_robots[0])
    masks, boxes, scores = output_robots["masks"], output_robots["boxes"], output_robots["scores"]
    robot_mask = masks.detach().cpu().numpy()
    robot_mask = np.any(robot_mask, axis=0).squeeze(0)

    for robot_prompt in prompts_robots:
        output_robots = processor.set_text_prompt(state=inference_state, prompt= robot_prompt)
        masks, boxes, scores = output_robots["masks"], output_robots["boxes"], output_robots["scores"]
        print(robot_prompt + "   " + str(masks.shape[0]))
        mask_np = masks.detach().cpu().numpy()
        mask_np = np.any(mask_np, axis=0).squeeze(0)
        robot_mask = mask_np | robot_mask

        """
        plt.imshow(mask_np, cmap='gray')
        plt.title("Mask")
        plt.show()
        plt.imsave(folder_dir + image_path + line_prompt + "_mask.png", mask_np, cmap='gray')
        """
    
    print("")

    #########################################For field##########################################
    prompts_field = ["football field", "field", "grass field"]

    print("************************")
    print("field detection")
    print("************************")

    output_field = processor.set_text_prompt(state=inference_state, prompt= prompts_field[0])
    masks, boxes, scores = output_field["masks"], output_field["boxes"], output_field["scores"]
    field_mask = masks.detach().cpu().numpy()
    field_mask = np.any(field_mask, axis=0).squeeze(0)

    for field_prompt in prompts_field:
        output_field = processor.set_text_prompt(state=inference_state, prompt= field_prompt)
        masks, boxes, scores = output_field["masks"], output_field["boxes"], output_field["scores"]
        print(field_prompt + "   " + str(masks.shape[0]))
        mask_np = masks.detach().cpu().numpy()
        mask_np = np.any(mask_np, axis=0).squeeze(0)
        field_mask = mask_np | field_mask

    print("")

    #########################################for Ball###########################################
    prompts_ball = ["football", "soccer ball"]

    print("************************")
    print("ball detection")
    print("************************")

    output_ball = processor.set_text_prompt(state=inference_state, prompt= prompts_ball[0])
    masks, boxes, scores = output_ball["masks"], output_ball["boxes"], output_ball["scores"]
    ball_mask = masks.detach().cpu().numpy()
    ball_mask = np.any(ball_mask, axis=0).squeeze(0)

    for ball_prompt in prompts_ball:
        output_ball = processor.set_text_prompt(state=inference_state, prompt= ball_prompt)
        masks, boxes, scores = output_ball["masks"], output_ball["boxes"], output_ball["scores"]
        print(ball_prompt + "   " + str(masks.shape[0]))
        mask_np = masks.detach().cpu().numpy()
        mask_np = np.any(mask_np, axis=0).squeeze(0)
        ball_mask = mask_np | ball_mask
    print("")
    image_current +=1    
    print(f"Progress: {image_current} / {images_amount}")

    #########################################Masks Combined#####################################         

    red_map = ListedColormap([[0,0,0,0],[1,0,0,1]])
    blue_map = ListedColormap([[0,0,0,0],[0,0,1,0.6]])
    green_map = ListedColormap([[0,0,0,0],[0,1,0,0.3]])
    orange_map = ListedColormap([[0,0,0,0],[1,0.5,0,0.8]])

    red_mark = patches.Patch(color='red', label='line')
    blue_mark = patches.Patch(color='blue', label = 'robot')
    green_mark = patches.Patch(color = 'green', label = 'field')
    orange_mark = patches.Patch(color = 'orange', label = 'ball')

    plt.legend(handles = [red_mark,blue_mark,green_mark,orange_mark])
   
    plt.imshow(rgb_image)
    plt.imshow(line_mask, cmap = red_map)
    plt.imshow(robot_mask, cmap = blue_map)
    plt.imshow(field_mask, cmap=green_map)
    plt.imshow(ball_mask, cmap = orange_map)
    plt.axis('off')
    #plt.title("Mask")    
    plt.show()
    plt.savefig(image_entry.path +  "_mask.png",bbox_inches='tight',pad_inches=0)
    #plt.imsave(folder_dir + image_path +  "_mask.png", comb_mask, cmap='gray')