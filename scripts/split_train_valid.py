import random

import yaml

validation = {"images": {}}

with open("data/reality_rfdetr/train/annotations.yaml") as f:
    train = yaml.safe_load(f)

print(f"Training images before: {len(train['images'])}")

# Randomly sample 10% of the training data for validation
for image_name, image_data in list(train["images"].items()):
    if random.random() < 0.1:
        validation["images"][image_name] = image_data
        del train["images"][image_name]

print(f"Training images after: {len(train['images'])}")
print(f"Validation images: {len(validation['images'])}")

with open("data/reality_rfdetr/train/annotations.yaml", "w") as f:
    yaml.safe_dump(train, f)

with open("data/reality_rfdetr/valid/annotations.yaml", "w") as f:
    yaml.safe_dump(validation, f)

print("Training and validation split completed.")
