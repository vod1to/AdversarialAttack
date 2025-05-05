
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from Utils.matlab_cp2tform import get_similarity_transform_for_cv2
model = "Facenet" #INPUT YOUR MODEL NAME OPTIONS: "Ada" "Sphere" "VGG" "Facenet"
landmark = {}
with open('Model/lfw_landmark/lfw_landmark.txt') as f:
        landmark_lines = f.readlines()
for line in landmark_lines:
        l = line.replace('\n','').split('\t')
        landmark[l[0]] = [int(k) for k in l[1:]]
def alignment(src_img,src_pts):
        ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
            [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
        crop_size = (112, 112)
        src_pts = np.array(src_pts).reshape(5,2)

        s = np.array(src_pts).astype(np.float32)
        r = np.array(ref_pts).astype(np.float32)

        tfm = get_similarity_transform_for_cv2(s, r)
        face_img = cv2.warpAffine(src_img, tfm, crop_size)
        return face_img
import os
def load_images(original_path, adversarial_path):
    """Load both original and adversarial images."""
    if model == "Ada" or model == "Sphere":
        original = cv2.imread(original_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        original_parts = original_path.split(os.sep)
        person1 = original_parts[-2]
        file1 = original_parts[-1]
        landmark_key1 = f"{person1}/{file1}"
        adversarial = np.array(Image.open(adversarial_path))
        original = alignment(original, landmark[landmark_key1])
    else:
        original = np.array(Image.open(original_path))
    adversarial = np.array(Image.open(adversarial_path))

    # Convert to same shape and type if needed
    if original.shape != adversarial.shape:
        adversarial = cv2.resize(adversarial, (original.shape[1], original.shape[0]))
    
    if original.dtype != adversarial.dtype:
        # Convert to same data type
        adversarial = adversarial.astype(original.dtype)
        
    return original, adversarial

def generate_difference_heatmap(original, adversarial):
    """Generate a heatmap showing the difference between original and adversarial images."""
    # Calculate absolute pixel-wise difference
    diff = np.abs(original.astype(np.float32) - adversarial.astype(np.float32))
    
    # If images are RGB, convert difference to grayscale for clearer visualization
    if len(diff.shape) == 3 and diff.shape[2] == 3:
        diff_gray = np.mean(diff, axis=2)
    else:
        diff_gray = diff
        
    # Normalize the difference for better visualization
    if np.max(diff_gray) > 0:
        diff_normalized = diff_gray / np.max(diff_gray)
    else:
        diff_normalized = diff_gray
        
    return diff_normalized

def visualize_attack(original_path, adversarial_path, target_path=None, save_path=None):
    """Display the images in a 2x2 grid: original and adversarial on top row,
       target and difference heatmap on bottom row."""
    # Load images
    original, adversarial = load_images(original_path, adversarial_path)
    
    # Generate difference heatmap
    diff_heatmap = generate_difference_heatmap(original, adversarial)
    
    # Create a 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plot original image (top left)
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Plot adversarial image (top right)
    axes[0, 1].imshow(adversarial)
    axes[0, 1].set_title('Adversarial Image')
    axes[0, 1].axis('off')
    
    # Plot target image if provided (bottom left)
    if target_path:
        if model == "Ada" or model == "Sphere":
            target = cv2.imread(target_path)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            original2_parts = target_path.split(os.sep)
            person2 = original2_parts[-2]
            file2 = original2_parts[-1]
            landmark_key1 = f"{person2}/{file2}"
            target = alignment(target, landmark[landmark_key1])
        else:
            target = np.array(Image.open(target_path))
        # Resize if needed
        if target.shape != original.shape:
            target = cv2.resize(target, (original.shape[1], original.shape[0]))
        axes[1, 0].imshow(target)
        axes[1, 0].set_title('Target Image')
        axes[1, 0].axis('off')
    else:
        # If no target, hide this subplot
        axes[1, 0].axis('off')
        axes[1, 0].set_title('No Target Image')
    
    # Plot difference heatmap (bottom right)
    heatmap = axes[1, 1].imshow(diff_heatmap, cmap='hot')
    axes[1, 1].set_title('Difference Heatmap')
    axes[1, 1].axis('off')
    
    # Add colorbar for heatmap
    cbar = fig.colorbar(heatmap, ax=axes[1, 1], orientation='vertical')
    cbar.set_label('Magnitude of Perturbation')
    
    plt.tight_layout()
    
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Also print some statistics about the perturbation
    diff = np.abs(original.astype(np.float32) - adversarial.astype(np.float32))
    print(f"Maximum pixel difference: {np.max(diff)}")
    print(f"Average pixel difference: {np.mean(diff)}")
    print(f"Percentage of modified pixels (>0): {100 * np.sum(diff > 0) / diff.size:.2f}%")

# Example usage
if __name__ == "__main__":
    # Replace these with your actual image paths
    original_image_path = "E:\lfw\lfw-py\lfw_funneled\Aaron_Peirsol\Aaron_Peirsol_0001.jpg"
    adversarial_image_path = "E:\lfw\lfw-py\lfw_funneled\Aaron_Peirsol\Aaron_Peirsol_0001.jpg"
    target_image_path =  "E:\lfw\lfw-py\lfw_funneled\Aaron_Peirsol\Aaron_Peirsol_0002.jpg" # Optional, set to None if not available
    
    # Call with target image
    visualize_attack(
        original_image_path, 
        adversarial_image_path, 
        target_image_path, 
        "heatmap_result.png"
    )