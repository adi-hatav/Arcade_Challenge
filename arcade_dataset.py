import json
import os

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class ArcadeDataset(Dataset):
    def __init__(
        self, image_dir, annotation_file, transform_dirs, transform=None, num_classes=25
    ):
        self.num_classes = num_classes
        self.image_dir = image_dir
        self.transform_dirs = transform_dirs
        self.transform = transform
        self.image_files = os.listdir(image_dir)
        self.json = json.load(open(annotation_file))
        self.annotations = self.json["annotations"]
        self.img_id2file = {}
        for rec in self.json['images']:
            img_id, img_file = rec['id'], rec['file_name']
            self.img_id2file[img_id] = img_file

    def __len__(self):
        return len(self.image_files)

    def create_masks_and_labels(self, image_id, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        separate_masks = np.zeros((self.num_classes, height, width), dtype=np.uint8)
        labels = np.zeros(self.num_classes, dtype=np.uint8)
        for annot in self.annotations:
            if annot["image_id"] == image_id:
                category_id = (
                    annot["category_id"] - 1
                )  # -1 because category_id starts from 1
                labels[category_id] = 1
                for seg in annot["segmentation"]:
                    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], color=1)
                    cv2.fillPoly(separate_masks[category_id], [pts], color=1)
        return mask, separate_masks, labels

    def load_transformed_image(self, img_file, transform_type):
        transform_img_path = os.path.join(
            self.transform_dirs[transform_type], "images", img_file
        )
        if os.path.exists(transform_img_path):
            return cv2.imread(transform_img_path, cv2.IMREAD_GRAYSCALE)
        raise FileNotFoundError(f"Transformed image not found at {transform_img_path}")

    def __getitem__(self, idx):
        img_name = self.img_id2file[idx + 1]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        height, width = image.shape

        # Apply transformation
        transformed_image_top_hat = self.load_transformed_image(
            img_name, "top_hat_transform"
        )
        transformed_image_canny = self.load_transformed_image(img_name, "canny_edge")
        transformed_image = np.stack(
            (image, transformed_image_top_hat, transformed_image_canny), axis=0
        )

        # Create masks and labels
        image_id = idx + 1
        mask, separate_masks, labels = self.create_masks_and_labels(
            image_id, height, width
        )

        return {
            "original_image": torch.tensor(image, dtype=torch.float32).unsqueeze(0),
            "transformed_image": torch.tensor(transformed_image, dtype=torch.float32),
            "masks": torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
            "separate_masks": torch.tensor(separate_masks, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


def load_dataset(split, base_path="data/arcade/syntax", transform=None, num_classes=25):
    # Define paths for the specific split
    image_dir = os.path.join(base_path, split, "images")
    annotation_file = os.path.join(base_path, split, "annotations", f"{split}.json")

    transform_dirs = {
        "top_hat_transform": os.path.join(
            "data", "arcade", "processed", "syntax", "top_hat_transform", split
        ),
        "canny_edge": os.path.join(
            "data", "arcade", "processed", "syntax", "canny_edge", split
        ),
    }

    dataset = ArcadeDataset(
        image_dir=image_dir,
        annotation_file=annotation_file,
        transform_dirs=transform_dirs,
        transform=transform,
        num_classes=num_classes,
    )
    return dataset


# Custom colors for classes
custom_colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
]


def get_color(category_id):
    return custom_colors[category_id % len(custom_colors)]


# Visualize the images and masks
def visualize_batch(batch, num_classes=25, num_images=1):
    fig, axes = plt.subplots(5, num_images, figsize=(20, 25))

    # Titles for each row
    row_titles = [
        "Original Image",
        "Transformed Image (Top Hat)",
        "Transformed Image (Canny Edge)",
        "Mask",
        "Separate Masks",
    ]

    for row_idx, row_title in enumerate(row_titles):
        # Display row title
        axes[row_idx, 0].set_ylabel(row_title, fontsize=16, labelpad=20)

        for i in range(num_images):
            ax = axes[row_idx, i]
            if row_idx == 0:  # Original Image
                ax.imshow(
                    batch["original_image"][i].squeeze().cpu().numpy(), cmap="gray"
                )
            elif row_idx == 1:  # Transformed Image (Top Hat)
                ax.imshow(
                    batch["transformed_image"][i]
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()[:, :, 1],
                    cmap="gray",
                )
            elif row_idx == 2:  # Transformed Image (Canny Edge)
                ax.imshow(
                    batch["transformed_image"][i]
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()[:, :, 2],
                    cmap="gray",
                )
            elif row_idx == 3:  # Mask
                ax.imshow(batch["masks"][i].squeeze().cpu().numpy(), cmap="gray")
            else:  # Separate Masks
                ax.imshow(
                    batch["original_image"][i].squeeze().cpu().numpy(), cmap="gray"
                )
                for class_id in range(num_classes):
                    mask = batch["separate_masks"][i, class_id].cpu().numpy()
                    if np.any(mask):
                        color = get_color(class_id)
                        contours, _ = cv2.findContours(
                            mask.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        for contour in contours:
                            ax.add_patch(
                                patches.Polygon(
                                    contour.squeeze(),
                                    closed=True,
                                    fill=True,
                                    edgecolor=color,
                                    facecolor=color,
                                    alpha=0.3,
                                )
                            )
                            x, y, w, h = cv2.boundingRect(contour)
                            ax.add_patch(
                                patches.Rectangle(
                                    (x, y),
                                    w,
                                    h,
                                    linewidth=0.8,
                                    edgecolor=color,
                                    facecolor="none",
                                )
                            )
                            ax.text(
                                x, y, str(class_id), fontsize=12, alpha=0.7, color=color
                            )

            # Add labels to the title of each image
            if row_idx == 0:
                labels = batch["labels"][i].cpu().numpy()
                label_indices = np.where(labels == 1)[0]
                ax.set_title(
                    f"Labels: {', '.join(map(str, label_indices))}", fontsize=12
                )

            ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
        load_dataset(split="train"), batch_size=1
    )
    for batch in train_loader:
        print(
            batch["original_image"].shape,
            batch["transformed_image"].shape,
            batch["masks"].shape,
            batch["separate_masks"].shape,
        )
        break
