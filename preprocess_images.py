import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def top_hat_transform(image):
    # Apply the white top-hat transformation
    neg_image = cv2.bitwise_not(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    top_hat = cv2.morphologyEx(neg_image, cv2.MORPH_TOPHAT, kernel)

    # Subtract the top-hat result from the original image
    subtracted = cv2.subtract(image, top_hat)

    # Further reduce noise using a Bilateral Filter
    denoised = cv2.bilateralFilter(subtracted, 5, 75, 75)

    # Clip the result to the range [0, 255]
    clipped = np.clip(denoised, 0, 255).astype(np.uint8)

    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(clipped)

    return enhanced_image

def canny_edge(image):
    # Apply the white top-hat transformation
    neg_image = cv2.bitwise_not(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    top_hat = cv2.morphologyEx(neg_image, cv2.MORPH_TOPHAT, kernel)

    # Subtract the top-hat result from the original image
    subtracted = cv2.subtract(image, top_hat)

    # Remove the noise and keep the edges using a canny edge detector
    edges = cv2.Canny(subtracted, 100, 230)

    # Clip the result to the range [0, 255]
    clipped = np.clip(edges, 0, 255).astype(np.uint8)

    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(clipped)

    return enhanced_image

def preprocess_all_imgs(transform):
    img_dir_prefix = 'data/arcade/syntax'
    out_dir_prefix = f'data/arcade/processed/syntax/{transform.__name__}'
    data_folds = ['train', 'val', 'test']

    if not os.path.exists(out_dir_prefix):
        os.makedirs(out_dir_prefix)

    for fold in data_folds:
        img_dir = os.path.join(img_dir_prefix, fold + '/images')
        out_dir = os.path.join(out_dir_prefix, fold)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        img_files = os.listdir(img_dir)
        for img_file in tqdm(img_files, desc=f'Processing {fold} images', unit='images'):
            img_path = os.path.join(img_dir, img_file)
            out_path = os.path.join(out_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = transform(img)
            cv2.imwrite(out_path, img)

def test_preprocess(transform, img_idx = None):
    if img_idx is None:
        img_idx = np.random.randint(1, 200)

    original_img_path = f'data/arcade/syntax/val/images/{img_idx}.png'
    processed_img_path = f'data/arcade/syntax/processed/{transform.__name__}/val/images/{img_idx}.png'

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    original_img = cv2.imread(original_img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    processed_img = cv2.imread(processed_img_path)
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    plt.imshow(processed_img, cmap='gray')
    plt.title('Processed Image')

    plt.show()

if __name__ == '__main__':
    preprocess_all_imgs(top_hat_transform)
    preprocess_all_imgs(canny_edge)
