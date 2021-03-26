import os
import glob
from PIL import Image
from torch.utils.data import Dataset


class ImageFolderWithPaths(Dataset):
    def __init__(self, folder_path, transform=None):
        files = sorted(glob.glob(folder_path + '/**/*', recursive=True))
        self.files = list(filter(
            lambda x: os.path.splitext(os.path.basename(x))[-1] in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"], 
            files))

        self.transform = transform

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img, img_path

    def __len__(self):
        return len(self.files)