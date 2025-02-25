import cv2
import yaml
import os
import argparse

# Global variable to store the clicked point in the cropped image coordinate system.
clicked_point = None

def on_mouse(event, x, y, flags, param):
    """ Callback function to capture mouse clicks on the displayed image. """
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

def process_yaml(yaml_file, output_yaml):
    global clicked_point

    # Get the directory of the YAML file
    yaml_dir = os.path.dirname(yaml_file)

    # Construct the image directory path
    image_dir = os.path.join(yaml_dir, "images")

    # Load the YAML file
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    # Process each image in the YAML
    for image_filename, image_info in data["images"].items():
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

            # Compute the bounding box
            x_min, y_min, x_max, y_max = get_bounding_box(annotation["vector"])

            # Crop the robot region
            cropped_robot = image[y_min:y_max, x_min:x_max].copy()
            if cropped_robot.size == 0:
                print("Empty crop encountered. Skipping this annotation.")
                continue

            # Reset the clicked point for this robot
            clicked_point = None

            # Create a window and set up the mouse callback
            window_name = "Robot Detection"
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, on_mouse)

            print(f"Displaying robot in {image_filename}.")
            print("Click on the base footprint of the robot, press 's' if no base footprint is visible,")
            print("press 'q' to exit without saving changes, or press 'w' to save and exit.")

            # Display loop
            while True:
                cv2.imshow(window_name, cropped_robot)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("Exiting without saving.")
                    cv2.destroyAllWindows()
                    return None  # Exit function without saving

                if key == ord("w"):
                    print("Saving and exiting.")
                    cv2.destroyAllWindows()
                    with open(output_yaml, "w") as f:
                        yaml.dump(data, f)
                    print(f"Updated YAML saved as {output_yaml}")
                    return None  # Exit function after saving

                # If the user presses 's', assign None to base_footprint
                if key == ord("s"):
                    annotation["base_footprint"] = None
                    print("User pressed 's'. Saving base_footprint as None.")
                    break

                # If a click has been recorded, transform the coordinates
                if clicked_point is not None:
                    # clicked_point is in cropped image coordinates
                    base_x = clicked_point[0] + x_min
                    base_y = clicked_point[1] + y_min
                    annotation["base_footprint"] = (base_x, base_y)
                    print(f"Recorded base_footprint at source image coordinate: ({base_x}, {base_y})")
                    break

            cv2.destroyWindow(window_name)

    # Save the updated YAML file
    with open(output_yaml, "w") as f:
        yaml.dump(data, f)

    print(f"Updated YAML saved as {output_yaml}")

def main():
    parser = argparse.ArgumentParser(description="Annotate robot base footprints in images based on YAML metadata.")
    parser.add_argument("yaml_file", help="Path to the YAML file containing image annotations.")
    parser.add_argument("output_yaml", help="Path to save the updated YAML file.")
    args = parser.parse_args()

    if not os.path.exists(args.yaml_file):
        print(f"Error: YAML file '{args.yaml_file}' not found.")
        return

    process_yaml(args.yaml_file, args.output_yaml)

if __name__ == "__main__":
    main()
