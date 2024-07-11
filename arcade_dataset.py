import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from preprocess_images import top_hat_transform, canny_edge


class ArcadeDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform_dirs, transform=None, num_classes=25):
        self.num_classes = num_classes
        self.image_dir = image_dir
        self.transform_dirs = transform_dirs
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.annotations = json.load(open(annotation_file))['annotations']

    def __len__(self):
        return len(self.image_files)

    def create_masks(self, image_id, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        separate_masks = np.zeros((self.num_classes, height, width), dtype=np.uint8)
        for annot in self.annotations:
            if annot['image_id'] == image_id:
                category_id = annot['category_id']
                for seg in annot['segmentation']:
                    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], color=1)
                    cv2.fillPoly(separate_masks[category_id-1], [pts], color=1) # -1 because category_id starts from 1
        return mask, separate_masks

    def load_transformed_image(self, img_name, transform_type):
        transform_img_path = os.path.join(self.transform_dirs[transform_type], img_name)
        if os.path.exists(transform_img_path):
            return cv2.imread(transform_img_path, cv2.IMREAD_GRAYSCALE)
        return None

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        height, width = image.shape

        # Apply transformation
        transformed_image_top_hat = self.load_transformed_image(img_name, 'top_hat_transform')
        transformed_image_canny = self.load_transformed_image(img_name, 'canny_edge')

        if transformed_image_top_hat is None:
            transformed_image_top_hat = self.transform(image)
        if transformed_image_canny is None:
            transformed_image_canny = self.transform(image)

        transformed_image = np.stack((image, transformed_image_top_hat, transformed_image_canny), axis=0)

        # Create masks
        image_id = idx + 1
        mask, separate_masks = self.create_masks(image_id, height, width)

        return {
            'original_image': torch.tensor(image, dtype=torch.float32).unsqueeze(0),
            'transformed_image': torch.tensor(transformed_image, dtype=torch.float32),
            'masks': torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
            'separate_masks': torch.tensor(separate_masks, dtype=torch.float32)
        }

def load_dataset(split, shuffle, base_path='data/arcade/syntax', batch_size=8, num_workers=4, transform=None,
                 num_classes=25):
    # Define paths for the specific split
    image_dir = os.path.join(base_path, split, 'images')
    annotation_file = os.path.join(base_path, split, 'annotations', f'{split}.json')

    transform_dirs = {
        'top_hat_transform': os.path.join('data', 'arcade', 'processed', 'syntax', 'top_hat_transform', split),
        'canny_edge': os.path.join('data', 'arcade', 'processed', 'syntax', 'canny_edge', split)
    }

    dataset = ArcadeDataset(
        image_dir=image_dir,
        annotation_file=annotation_file,
        transform_dirs=transform_dirs,
        transform=transform,
        num_classes=num_classes
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader
