import argparse
import os

import cv2
import numpy as np
import yaml

# Global variable to store the clicked point in the scaled image coordinate system.
clicked_point = None


def on_mouse(event, x, y, flags, param):
    """Callback function to capture mouse clicks on the displayed image."""
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)


def get_bounding_box(vector):
    """
    Given an unsorted list of points (each point is a two-element list),
    return the bounding box as (x_min, y_min, x_max, y_max).
    """
    xs = [pt[0] for pt in vector]
    ys = [pt[1] for pt in vector]
    return min(xs), min(ys), max(xs), max(ys)


def show_wait(window_name: str):
    """Display a wait message on the image window."""
    img = np.zeros((800, 800, 3), dtype=np.uint8)
    cv2.putText(
        img, "Wait...", (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )
    cv2.imshow(window_name, img)
    cv2.waitKey(50)
    return


def process_yaml(yaml_file, output_yaml, start_image):
    global clicked_point

    # Get the directory of the YAML file
    yaml_dir = os.path.dirname(yaml_file)

    # Construct the image directory path
    image_dir = os.path.join(yaml_dir, "images")

    # Create a window and set up the mouse callback
    main_window_name = "Robot Detection"
    cv2.namedWindow(main_window_name)
    cv2.setMouseCallback(main_window_name, on_mouse)

    # Create the context window
    context_window_name = "Context"
    cv2.namedWindow(context_window_name)

    # Load the YAML file
    show_wait(main_window_name)
    show_wait(context_window_name)
    with open(yaml_file, "r") as f:
        print(f"Loading YAML file {yaml_file}, this may take a moment...")
        data = yaml.safe_load(f)

    # Get the list of images in the YAML file
    image_keys = list(data["images"].keys())

    # Estimate the number of annotations to be done
    num_annotations_total = 0
    num_annotations_done = 0
    for image_filename in image_keys:
        image_info = data["images"][image_filename]
        for annotation in image_info.get("annotations", []):
            if annotation.get("type") != "robot":
                continue
            if "vector" not in annotation or not annotation["vector"]:
                continue
            num_annotations_total += 1
            if "base_footprint" in annotation:
                num_annotations_done += 1

    # Process each image in the YAML
    for i in range(start_image, len(image_keys)):
        # Get the image filename and image info
        image_filename = image_keys[i]
        image_info = data["images"][image_filename]

        # Construct full path to the image in the "images/" folder
        image_path = os.path.join(image_dir, image_filename)

        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"Image file {image_path} not found. Skipping.")
            continue

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image {image_path}. Skipping.")
            continue

        # Process each annotation
        for annotation in image_info.get("annotations", []):
            # Process only robot annotations
            if annotation.get("type") != "robot":
                continue

            # Skip robots that already have a base footprint annotation
            if "base_footprint" in annotation:
                print("Skipping robot with existing base footprint.")
                continue

            # Ensure the vector is valid
            if "vector" not in annotation or not annotation["vector"]:
                print("Skipping robot annotation without a valid vector.")
                continue

            # Compute the bounding box in the original image
            x_min, y_min, x_max, y_max = get_bounding_box(annotation["vector"])

            # Crop the robot region from the original image
            cropped_robot = image[y_min:y_max, x_min:x_max].copy()
            if cropped_robot.size == 0:
                print("Empty crop encountered. Skipping this annotation.")
                continue

            # Get original dimensions of the cropped region
            orig_height, orig_width = cropped_robot.shape[:2]

            # Scale the cropped robot region to a height of 800px while maintaining aspect ratio
            target_height = 800
            scale_factor = target_height / orig_height
            new_width = int(orig_width * scale_factor)
            resized_robot = cv2.resize(cropped_robot, (new_width, target_height))

            # Reset the clicked point for this robot
            clicked_point = None

            # Increment the number of annotations done
            num_annotations_done += 1

            print(f"Displaying robot in {image_filename}.")
            print(f"Annotation {num_annotations_done} of {num_annotations_total}.")
            print(
                "Click on the base footprint of the robot, press 's' if no base footprint is visible,"
            )
            print(
                "press 'q' to exit without saving changes, or press 'w' to save and exit."
            )

            # Show context window
            display_height = 400
            display_width = int(image.shape[1] * display_height / image.shape[0])
            context_img = cv2.rectangle(
                image.copy(), (x_min, y_min), (x_max, y_max), (0, 255, 0), 2
            )
            context_img = cv2.resize(context_img, (display_width, display_height))
            cv2.imshow(context_window_name, context_img)

            # Display loop
            while True:
                cv2.imshow(main_window_name, resized_robot)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("Exiting without saving.")
                    cv2.destroyAllWindows()
                    return None  # Exit function without saving

                if key == ord("w"):
                    print("Saving and exiting.")
                    show_wait(
                        main_window_name
                    )  # Signal the user that we are not finished yet
                    show_wait(context_window_name)
                    with open(output_yaml, "w") as f:
                        yaml.dump(data, f)
                    cv2.destroyAllWindows()
                    print(f"Updated YAML saved as {output_yaml}")
                    return None  # Exit function after saving

                # If the user presses 's', assign None to base_footprint
                if key == ord("s"):
                    annotation["base_footprint"] = None
                    print("User pressed 's'. Saving base_footprint as None.")
                    break

                # If a click has been recorded, transform the coordinates
                if clicked_point is not None:
                    # Transform back to original image coordinates
                    base_x = int(clicked_point[0] / scale_factor) + x_min
                    base_y = int(clicked_point[1] / scale_factor) + y_min

                    annotation["base_footprint"] = [base_x, base_y]
                    print(
                        f"Recorded base_footprint at source image coordinate: ({base_x}, {base_y})"
                    )
                    break

    # Signal the user that we are not finished yet
    show_wait(main_window_name)
    show_wait(context_window_name)

    # Save the updated YAML file
    with open(output_yaml, "w") as f:
        yaml.dump(data, f)

    cv2.destroyAllWindows()

    print(f"Updated YAML saved as {output_yaml}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate robot base footprints in images based on YAML metadata."
    )
    parser.add_argument(
        "yaml_file", help="Path to the YAML file containing image annotations."
    )
    parser.add_argument("output_yaml", help="Path to save the updated YAML file.")
    parser.add_argument(
        "--start_image",
        "-n",
        type=int,
        default=0,
        help="Start from the Nth image in the YAML file.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.yaml_file):
        print(f"Error: YAML file '{args.yaml_file}' not found.")
        return

    process_yaml(args.yaml_file, args.output_yaml, args.start_image)


if __name__ == "__main__":
    main()
