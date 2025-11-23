import argparse
import os

import cv2
import numpy as np
import yaml
from tqdm import tqdm


def draw_annotations(image, annotations):
    """Draws bounding boxes and base footprint points on the image."""
    for annotation in annotations:
        if annotation.get("type") != "robot":
            continue

        # Check if annotation has a vector
        if "vector" in annotation:
            x_min, y_min = (
                min(p[0] for p in annotation["vector"]),
                min(p[1] for p in annotation["vector"]),
            )
            x_max, y_max = (
                max(p[0] for p in annotation["vector"]),
                max(p[1] for p in annotation["vector"]),
            )

            # Determine color based on base_footprint status
            if "base_footprint" in annotation:
                color = (
                    (0, 255, 255)
                    if annotation["base_footprint"] is None
                    else (0, 255, 0)
                )  # Yellow / Green
            else:
                color = (0, 0, 255)  # Red (no base footprint info)

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # Draw base footprint only if it's defined
        if "base_footprint" in annotation and annotation["base_footprint"] is not None:
            base_x, base_y = annotation["base_footprint"]
            cv2.circle(image, (base_x, base_y), 5, (0, 0, 255), -1)
            cv2.putText(
                image,
                "Base",
                (base_x + 5, base_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )


def save_yaml(yaml_file, data):
    """Saves the YAML file and displays a saving message."""
    saving_screen = 255 * np.ones((200, 400, 3), dtype=np.uint8)
    cv2.putText(
        saving_screen,
        "Saving...",
        (120, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )
    cv2.imshow("Saving", saving_screen)
    cv2.waitKey(500)
    with open(yaml_file, "w") as f:
        yaml.dump(data, f)
    cv2.destroyWindow("Saving")
    print(f"Updated YAML saved as {yaml_file}")


def visualize_annotations(yaml_file, start_image=None):
    # Load YAML file
    print(f"Loading YAML file {yaml_file}. This may take a moment...")
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    image_dir = os.path.join(os.path.dirname(yaml_file), "images")
    image_keys = list(data["images"].keys())

    if start_image and start_image in image_keys:
        start_index = image_keys.index(start_image)
    else:
        start_index = 0

    for i in tqdm(range(0, len(image_keys))):
        # Skip images before the start image, but still loop so the progress bar is accurate
        if i < start_index:
            continue

        # Get the data
        image_filename = image_keys[i]
        image_info = data["images"][image_filename]

        # Skip images where no robots have a "vector"
        if not any(
            "vector" in a
            for a in image_info.get("annotations", [])
            if a.get("type") == "robot"
        ):
            continue

        image_path = os.path.join(image_dir, image_filename)
        if not os.path.exists(image_path):
            print(f"Image {image_path} not found. Skipping.")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image {image_path}. Skipping.")
            continue

        print(f"Current image: {image_filename}")

        while True:
            display_image = image.copy()
            draw_annotations(display_image, image_info.get("annotations", []))

            cv2.imshow("Annotations Viewer", display_image)
            key = cv2.waitKey(0) & 0xFF

            if key == ord(" "):
                break  # Next image
            elif key == ord("d"):
                # Delete all base_footprint annotations for robots
                for annotation in image_info.get("annotations", []):
                    if (
                        annotation.get("type") == "robot"
                        and "base_footprint" in annotation
                    ):
                        del annotation["base_footprint"]
                print(f"Cleared base footprints for {image_filename}.")
            elif key == ord("w"):
                print(f"Saving YAML file {yaml_file}. This may take a moment...")
                save_yaml(yaml_file, data)
            elif key == ord("q"):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and modify robot annotations from a YAML file."
    )
    parser.add_argument(
        "yaml_file", help="Path to the YAML file containing image annotations."
    )
    parser.add_argument(
        "--start_image", help="Filename of the image to start from.", default=None
    )
    args = parser.parse_args()

    if not os.path.exists(args.yaml_file):
        print(f"Error: YAML file '{args.yaml_file}' not found.")
        return

    visualize_annotations(args.yaml_file, args.start_image)


if __name__ == "__main__":
    main()
