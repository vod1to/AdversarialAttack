import numpy as np
from deepface import DeepFace
import cv2
from tqdm import tqdm
import os
import tensorflow as tf

class AttackVisualizer():
    def __init__(self, model_name):
        self.model_name = model_name
        
    def save_image(self, img_array, path):
        """Save image array to file"""
        if img_array.dtype != np.uint8:
            img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(path, img_array)
        return path

    def compute_embedding(self, img_path):            
        try:
            result = DeepFace.represent(
                img_path=img_path,
                model_name=self.model_name,
                max_faces=1
            )  
            if not result or len(result) == 0:
                return None
            embedding = np.array(result[0]['embedding'])
            return embedding
        except Exception as e:
            print(f"Error computing embedding: {e}")
            return None
    def compute_gradient_batch(self, img_path, target_embedding, label, batch_size=16):
        """Compute gradient more efficiently using batching"""
        epsilon = 1e-7
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = img.astype(np.float32) / 255.0
        gradient = np.zeros_like(img)
        
        # Get base embedding once
        base_path = self.save_image(img, 'base.jpg')
        base_embedding = self.compute_embedding(base_path)
        base_sim = self.compute_cosine_similarity(target_embedding, base_embedding)
        
        height, width = img.shape[:2]
        step = 1
        
        # Process pixels in batches
        for h_start in range(0, height, step * batch_size):
            h_end = min(h_start + step * batch_size, height)
            for w_start in range(0, width, step * batch_size):
                w_end = min(w_start + step * batch_size, width)
                
                # Create batch of perturbed images
                batch_img = np.tile(img[h_start:h_end, w_start:w_end], (3, 1, 1, 1))
                
                # Apply perturbations to batch
                for idx, channel in enumerate(range(3)):
                    batch_img[idx, :, :, channel] += epsilon
                
                # Process batch
                batch_results = []
                for b_idx in range(3):
                    temp_img = img.copy()
                    temp_img[h_start:h_end, w_start:w_end] = batch_img[b_idx]
                    temp_path = self.save_image(temp_img, f'batch_{b_idx}.jpg')
                    perturbed_embedding = self.compute_embedding(temp_path)
                    if perturbed_embedding is not None:
                        sim = self.compute_cosine_similarity(target_embedding, perturbed_embedding)
                        batch_results.append(sim)
                    os.remove(temp_path)
                
                # Compute gradients for batch
                if len(batch_results) == 3:
                    for h in range(h_start, h_end, step):
                        for w in range(w_start, w_end, step):
                            for c in range(3):
                                grad_val = (batch_results[c] - base_sim) / epsilon
                                if label == 1:
                                    grad_val = -grad_val
                                # Apply gradient to block
                                h_block = min(step, h_end - h)
                                w_block = min(step, w_end - w)
                                gradient[h:h+h_block, w:w+w_block, c] = grad_val
        
        os.remove(base_path)
        return gradient

    def apply_fgsm_attack(self, img_path, target_embedding, epsilon=0.5, label=None, save_visualization=False):
        img = cv2.imread(img_path)
        img = img.astype(np.float32) / 255.0
        
        gradient = self.compute_gradient_batch(img_path, target_embedding, label)
        perturbation = epsilon * np.sign(gradient)
        perturbed_img = img + perturbation
        perturbed_img = np.clip(perturbed_img, 0, 1)
        
        perturbed_path = self.save_image(perturbed_img, "temp_adv.jpg")
        
        if save_visualization:
            viz_outputs = self.visualize_perturbation(img_path, perturbed_path)
            os.remove(perturbed_path)
            return perturbed_path, viz_outputs
        
        return perturbed_path
    def compute_cosine_similarity(self, embedding1, embedding2):
        """Compute cosine similarity between two embeddings"""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(embedding1, embedding2) / (norm1 * norm2)
        
    def visualize_perturbation(self, original_img_path, perturbed_img_path, save_path="adv_images/perturbation.jpg"):
        """
        Visualize and save the perturbation layer.
        Returns paths to saved visualization images.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Read images
        original = cv2.imread(original_img_path).astype(np.float32) / 255.0
        perturbed = cv2.imread(perturbed_img_path).astype(np.float32) / 255.0
        
        # Calculate perturbation
        perturbation = perturbed - original
        
        # Create heatmap of perturbation magnitude
        perturbation_magnitude = np.sqrt(np.sum(perturbation**2, axis=-1))
        perturbation_magnitude = (perturbation_magnitude - np.min(perturbation_magnitude)) / \
                               (np.max(perturbation_magnitude) - np.min(perturbation_magnitude))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap((perturbation_magnitude * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Create enhanced perturbation visualization
        enhanced_perturbation = np.abs(perturbation) * 5  # Enhance the difference by scaling
        enhanced_perturbation = np.clip(enhanced_perturbation, 0, 1)
        
        # Save visualizations
        outputs = {}
        
        # Save original
        orig_path = save_path.replace('.jpg', '_original.jpg')
        cv2.imwrite(orig_path, cv2.imread(original_img_path))
        outputs['original'] = orig_path
        
        # Save perturbed
        pert_path = save_path.replace('.jpg', '_perturbed.jpg')
        cv2.imwrite(pert_path, cv2.imread(perturbed_img_path))
        outputs['perturbed'] = pert_path
        
        # Save enhanced perturbation
        enhanced_path = save_path.replace('.jpg', '_enhanced.jpg')
        cv2.imwrite(enhanced_path, (enhanced_perturbation * 255).astype(np.uint8))
        outputs['enhanced'] = enhanced_path
        
        # Save heatmap
        heatmap_path = save_path.replace('.jpg', '_heatmap.jpg')
        cv2.imwrite(heatmap_path, heatmap)
        outputs['heatmap'] = heatmap_path
        
        # Create side-by-side comparison
        h, w = original.shape[:2]
        comparison = np.zeros((h, w*4, 3), dtype=np.uint8)
        
        # Convert all images to uint8 for concatenation
        orig_uint8 = (original * 255).astype(np.uint8)
        pert_uint8 = (perturbed * 255).astype(np.uint8)
        enh_uint8 = (enhanced_perturbation * 255).astype(np.uint8)
        
        # Create side-by-side comparison
        comparison[:, :w] = orig_uint8
        comparison[:, w:2*w] = pert_uint8
        comparison[:, 2*w:3*w] = enh_uint8
        comparison[:, 3*w:] = heatmap
        
        # Save comparison
        comparison_path = save_path.replace('.jpg', '_comparison.jpg')
        cv2.imwrite(comparison_path, comparison)
        outputs['comparison'] = comparison_path
        
        return outputs
def main():
    visualizer = AttackVisualizer(model_name="VGG-Face")
    
    # Replace with your image paths
    img_path = "E:/lfw/lfw-py/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg"
    target_path = "E:/lfw/lfw-py/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0002.jpg"
    target_embedding = visualizer.compute_embedding(img_path=target_path)


    adv_img_path, viz_outputs = visualizer.apply_fgsm_attack(
        img_path, 
        target_embedding, 
        epsilon=0.5, 
        label=1, 
        save_visualization=True
    )

if __name__ == "__main__":
    main()