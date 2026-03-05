#!/usr/bin/env python3
"""
Compare images in a folder using OpenCV's Structural Similarity Index (SSIM).
Each image is compared with every other image in the folder.
Supports Fuji RAW (.RAF) files by either converting them or using embedded JPEGs.
"""

import cv2
import os
import numpy as np
import rawpy
import tempfile
from itertools import combinations


def compare_images_with_ssim(image1, image2):
    """
    Compare two images using SSIM (Structural Similarity Index).
    Returns the SSIM score between the two images.
    """
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Resize images to same dimensions if needed
    if gray1.shape != gray2.shape:
        height = min(gray1.shape[0], gray2.shape[0])
        width = min(gray1.shape[1], gray2.shape[1])
        gray1 = cv2.resize(gray1, (width, height))
        gray2 = cv2.resize(gray2, (width, height))
    
    # Simple SSIM implementation using OpenCV and numpy
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Compute means
    mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)
    
    # Compute variances and covariance
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2
    
    # Compute SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    # Avoid division by zero and ensure valid range
    denominator = np.maximum(denominator, 1e-8)
    ssim_map = numerator / denominator
    
    # Clip to [0, 1] range to handle numerical precision issues
    ssim_map = np.clip(ssim_map, 0, 1)
    
    return np.mean(ssim_map)


def convert_raw_to_tiff(raw_path, output_path):
    """
    Convert a RAW file to TIFF using rawpy.
    Returns True if successful, False otherwise.
    """
    try:
        with rawpy.imread(raw_path) as raw:
            # Post-process the RAW image (demosaicing, white balance, etc.)
            # rgb = raw.postprocess()
            print("Converting... ")
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=True,  # Faster processing, still good for similarity
                no_auto_bright=True,
                output_bps=8
            )

            # Convert to BGR for OpenCV
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # Save as TIFF
            cv2.imwrite(output_path, bgr)
        return True
    except Exception:
        return False


def extract_embedded_jpeg(raw_path, output_path):
    """
    Extract embedded JPEG from RAW file using rawpy.
    Returns True if successful, False otherwise.
    """
    try:
        with rawpy.imread(raw_path) as raw:
            # Get the embedded thumbnail (JPEG)
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                with open(output_path, 'wb') as f:
                    f.write(thumb.data)
                return True
        return False
    except Exception:
        return False


def load_images_from_folder(folder_path, raw_mode='convert'):
    """
    Load all images from a folder.
    Returns a list of tuples: (filename, image_data)
    
    Args:
        folder_path: Path to the folder containing images
        raw_mode: How to handle RAW files - 'convert' (to TIFF) or 'embedded' (use JPEG)
    """
    images = []
    temp_dir = tempfile.mkdtemp()
    
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        
        # Handle RAW files
        if filename.lower().endswith('.raf'):
            if raw_mode == 'convert':
                temp_path = os.path.join(temp_dir, f"{os.path.splitext(filename)[0]}.tiff")
                if convert_raw_to_tiff(filepath, temp_path):
                    image = cv2.imread(temp_path)
                    if image is not None:
                        images.append((filename, image))
            elif raw_mode == 'embedded':
                temp_path = os.path.join(temp_dir, f"{os.path.splitext(filename)[0]}.jpg")
                if extract_embedded_jpeg(filepath, temp_path):
                    image = cv2.imread(temp_path)
                    if image is not None:
                        images.append((filename, image))
        
        # Handle regular image files
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image = cv2.imread(filepath)
            if image is not None:
                images.append((filename, image))
    
    # Clean up temporary files
    for temp_file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, temp_file))
    os.rmdir(temp_dir)
    
    return images


def main():
    """
    Main function to compare images in a folder.
    """
    default_path = "/Users/ofloericke/images"
    folder_path = input(f"Enter the path to the folder containing images [default: {default_path}]: ")
    
    # Use default path if user just presses Enter
    folder_path = folder_path.strip() if folder_path.strip() else default_path
    
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return
    
    # Ask user for RAW handling preference
    print("How would you like to handle Fuji RAW (.RAF) files?")
    print("1. Convert to TIFF (higher quality, requires dcraw)")
    print("2. Use embedded JPEG (faster, lower quality, requires exiftool)")
    raw_choice = input("Enter your choice (1 or 2): ")
    
    raw_mode = 'convert' if raw_choice == '1' else 'embedded'
    
    images = load_images_from_folder(folder_path, raw_mode)
    
    if len(images) < 2:
        print("Error: At least two images are required for comparison.")
        return
    
    print(f"Loaded {len(images)} images. Starting comparisons...")
    
    # Compare each pair of images and store results
    results = []
    for (name1, img1), (name2, img2) in combinations(images, 2):
        ssim_score = compare_images_with_ssim(img1, img2)
        results.append((ssim_score, name1, name2))
    
    # Sort results by SSIM score (highest first)
    results.sort(reverse=True, key=lambda x: x[0])
    
    # Print sorted results
    print("\nComparison results sorted by SSIM score (highest similarity first):")
    for i, (score, name1, name2) in enumerate(results, 1):
        print(f"{i}. {name1} vs {name2} - SSIM Score: {score:.4f}")
    
    print(f"\nAll {len(results)} comparisons completed.")


if __name__ == "__main__":
    main()