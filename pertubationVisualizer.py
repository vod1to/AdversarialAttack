import numpy as np
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import os

class AttackVisualizer:
    def __init__(self, model_name="VGG-Face"):
        self.model_name = model_name
        
    def apply_attack(self, img_path, target_path, epsilon=0.3):
        """Apply FGSM attack and return original, perturbed, and perturbation"""
        # Read images
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = img.astype(np.float32) / 255.0
        
        # Get target embedding
        target_embedding = None
        try:
            result = DeepFace.represent(
                img_path=target_path,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='opencv'
            )
            if result and len(result) > 0:
                target_embedding = np.array(result[0]['embedding'])
        except Exception as e:
            print(f"Error computing target embedding: {e}")
            return None, None, None
            
        if target_embedding is None:
            return None, None, None
            
        # Compute gradient
        gradient = self.compute_gradient(img_path, target_embedding, label=1)
        
        # Apply perturbation
        perturbation = epsilon * np.sign(gradient)
        perturbed_img = img + perturbation
        perturbed_img = np.clip(perturbed_img, 0, 1)
        
        return img, perturbed_img, perturbation
        
    def compute_gradient(self, img_path, target_embedding, label):
        """Simplified gradient computation for visualization"""
        epsilon = 1e-5
        img = cv2.imread(img_path)
        img = img.astype(np.float32) / 255.0
        
        # Get base similarity
        base_path = self.save_image(img * 255, 'base.jpg')
        base_embedding = self.compute_embedding(base_path)
        if base_embedding is None:
            os.remove(base_path)
            return np.zeros_like(img)
            
        base_sim = self.compute_cosine_similarity(target_embedding, base_embedding)
        
        gradient = np.zeros_like(img)
        for channel in range(3):
            temp_img = img.copy()
            temp_img[:, :, channel] += epsilon
            
            temp_path = self.save_image(temp_img * 255, f'temp_{channel}.jpg')
            perturbed_embedding = self.compute_embedding(temp_path)
            
            if perturbed_embedding is not None:
                sim = self.compute_cosine_similarity(target_embedding, perturbed_embedding)
                gradient[:, :, channel] = (sim - base_sim) / epsilon
            
            os.remove(temp_path)
        
        os.remove(base_path)
        return -gradient if label == 1 else gradient
    
    def compute_embedding(self, img_path):
        """Compute facial embedding"""
        try:
            result = DeepFace.represent(
                img_path=img_path,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='opencv'
            )
            if result and len(result) > 0:
                return np.array(result[0]['embedding'])
        except Exception as e:
            print(f"Error computing embedding: {e}")
        return None
    
    def compute_cosine_similarity(self, embedding1, embedding2):
        """Compute cosine similarity between embeddings"""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    def save_image(self, img_array, path):
        """Save image array to file"""
        if img_array.dtype != np.uint8:
            img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(path, img_array)
        return path
        
    def visualize_attack(self, img_path, target_path, epsilon=0.3, save_path=None):
        """Visualize original, target, perturbed images and perturbation map"""
        # Apply attack
        original, perturbed, perturbation = self.apply_attack(img_path, target_path, epsilon)
        if original is None:
            print("Attack failed")
            return
            
        # Load target image
        target = cv2.imread(target_path)
        target = target.astype(np.float32) / 255.0
        
        # Create figure
        plt.figure(figsize=(20, 5))
        
        # Original image
        plt.subplot(141)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # Target image
        plt.subplot(142)
        plt.imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
        plt.title('Target Image')
        plt.axis('off')
        
        # Perturbed image
        plt.subplot(143)
        plt.imshow(cv2.cvtColor(perturbed, cv2.COLOR_BGR2RGB))
        plt.title('Perturbed Image')
        plt.axis('off')
        
        # Perturbation heatmap
        plt.subplot(144)
        perturbation_magnitude = np.linalg.norm(perturbation, axis=2)
        plt.imshow(perturbation_magnitude, cmap='hot')
        plt.colorbar(label='Perturbation Magnitude')
        plt.title('Perturbation Heatmap')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
            
        # Print similarity scores
        original_embedding = self.compute_embedding(img_path)
        target_embedding = self.compute_embedding(target_path)
        perturbed_path = self.save_image(perturbed * 255, 'temp_perturbed.jpg')
        perturbed_embedding = self.compute_embedding(perturbed_path)
        
        if all(x is not None for x in [original_embedding, target_embedding, perturbed_embedding]):
            orig_target_sim = self.compute_cosine_similarity(original_embedding, target_embedding)
            pert_target_sim = self.compute_cosine_similarity(perturbed_embedding, target_embedding)
            print(f"\nSimilarity Scores:")
            print(f"Original vs Target: {orig_target_sim:.4f}")
            print(f"Perturbed vs Target: {pert_target_sim:.4f}")
            print(f"Similarity Change: {pert_target_sim - orig_target_sim:.4f}")
        
        os.remove(perturbed_path)
        
        return original, perturbed, perturbation

def main():
    visualizer = AttackVisualizer(model_name="VGG-Face")
    
    # Replace with your image paths
    img_path = "E:/lfw/lfw-py/lfw_funneled\Aaron_Peirsol\Aaron_Peirsol_0001.jpg"
    target_path = "E:/lfw/lfw-py/lfw_funneled\Aaron_Peirsol\Aaron_Peirsol_0002.jpg"
    
    print("Visualizing attack...")
    visualizer.visualize_attack(
        img_path, 
        target_path,
        epsilon=0.3,
        save_path="attack_visualization.png"
    )

if __name__ == "__main__":
    main()