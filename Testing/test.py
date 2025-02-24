import numpy as np
from deepface import DeepFace
import cv2
from tqdm import tqdm
import os
import torch
class DeepFaceAttackFramework:
    def __init__(self, data_dir, model_name):
        self.data_dir = data_dir
        self.model_name = model_name
        self.pairs = self.prepare_pairs()
        self.embedding_cache = {}
        
    def prepare_pairs(self):
        pairs = []
        classes = [d for d in os.listdir(self.data_dir) 
                  if os.path.isdir(os.path.join(self.data_dir, d))]
        
        # Same person pairs
        for person in classes:
            person_dir = os.path.join(self.data_dir, person)
            images = os.listdir(person_dir)
            if len(images) >= 2:
                img1 = os.path.join(person_dir, images[0])
                img2 = os.path.join(person_dir, images[1])
                pairs.append((img1, img2, 1))
            if len(pairs) == 50:
                break
        
        # Different person pairs
        for i in range(len(classes)):
            for j in range(i + 1, min(i + 2, len(classes))):
                img1 = os.path.join(self.data_dir, classes[i], 
                                  os.listdir(os.path.join(self.data_dir, classes[i]))[0])
                img2 = os.path.join(self.data_dir, classes[j], 
                                  os.listdir(os.path.join(self.data_dir, classes[j]))[0])
                pairs.append((img1, img2, 0))
            if len(pairs) == 100:
                break
        return pairs

    def save_image(self, img_array, path):
        """Save image array to file"""
        if img_array.dtype != np.uint8:
            img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(path, img_array)
        return path

    def compute_embedding(self, img_path):
        if img_path in self.embedding_cache:
            return self.embedding_cache[img_path]
            
        try:
            result = DeepFace.represent(
                img_path=img_path,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='opencv',
                max_faces=1
            )
            
            if not result or len(result) == 0:
                return None
                
            embedding = np.array(result[0]['embedding'])
            self.embedding_cache[img_path] = embedding
            return embedding
            
        except Exception as e:
            print(f"Error computing embedding: {e}")
            return None

    def compute_cosine_similarity(self, embedding1, embedding2):
        """Compute cosine similarity between two embeddings"""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(embedding1, embedding2) / (norm1 * norm2)

    def compute_gradient_batch(self, img_path, target_embedding, label, batch_size=16):
        """Compute gradient accounting for DeepFace's preprocessing steps"""
        epsilon = 1e-5
        
        # Read image in BGR format
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Keep original BGR format, just normalize to [0,1]
        img = img.astype(np.float32) / 255.0
        gradient = np.zeros_like(img)
        
        # Get base embedding with all DeepFace's preprocessing steps
        base_path = self.save_image(img * 255, 'base.jpg')  # Save denormalized image
        base_embedding = self.compute_embedding(base_path)
        if base_embedding is None:
            os.remove(base_path)
            return np.zeros_like(img)
            
        base_sim = self.compute_cosine_similarity(target_embedding, base_embedding)
        
        height, width = img.shape[:2]
        step = 2
        
        # Process in batches while preserving BGR order
        for h_start in range(0, height, step * batch_size):
            h_end = min(h_start + step * batch_size, height)
            for w_start in range(0, width, step * batch_size):
                w_end = min(w_start + step * batch_size, width)
                
                # Create batch of perturbed images
                batch_results = []
                for channel in range(3):  # BGR order
                    temp_img = img.copy()
                    temp_img[h_start:h_end, w_start:w_end, channel] += epsilon
                    # Clip to valid range
                    temp_img = np.clip(temp_img, 0, 1)
                    
                    # Save denormalized image for DeepFace
                    temp_path = self.save_image(temp_img * 255, f'batch_{channel}.jpg')
                    
                    # Get embedding through DeepFace's full pipeline
                    perturbed_embedding = self.compute_embedding(temp_path)
                    
                    if perturbed_embedding is not None:
                        sim = self.compute_cosine_similarity(target_embedding, perturbed_embedding)
                        batch_results.append(sim)
                    else:
                        batch_results.append(base_sim)
                        
                    os.remove(temp_path)
                
                # Compute gradients for the batch
                if len(batch_results) == 3:
                    for h in range(h_start, h_end, step):
                        for w in range(w_start, w_end, step):
                            h_block = min(step, h_end - h)
                            w_block = min(step, w_end - w)
                            
                            for c in range(3):
                                grad_val = (batch_results[c] - base_sim) / epsilon
                                
                                if label == 1:
                                    grad_val = -grad_val
                                
                                gradient[h:h+h_block, w:w+w_block, c] = grad_val
        
        # Normalize gradient
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 0:
            gradient = gradient / grad_norm
            
        os.remove(base_path)
        return gradient

    def apply_fgsm_attack(self, img_path, target_embedding, epsilon=0.3, label=None):
        img = cv2.imread(img_path)
        img = img.astype(np.float32) / 255.0
        
        gradient = self.compute_gradient_batch(img_path, target_embedding, label)
        perturbation = epsilon * np.sign(gradient)
        perturbed_img = img + perturbation
        perturbed_img = np.clip(perturbed_img, 0, 1)
        
        return self.save_image(perturbed_img*225, "temp_adv.jpg")

    def apply_pgd_attack(self, img_path, target_embedding, epsilon=0.3, 
                        alpha=0.03, steps=10, label=None):
        img = cv2.imread(img_path)
        img = img.astype(np.float32) / 255.0
        perturbed_img = img.copy()
        
        for _ in range(steps):
            temp_path = self.save_image(perturbed_img, "temp_adv.jpg")
            gradient = self.compute_gradient_batch(temp_path, target_embedding, label)
            
            perturbation = alpha * np.sign(gradient)
            perturbed_img = perturbed_img + perturbation
            perturbed_img = np.clip(perturbed_img, img - epsilon, img + epsilon)
            perturbed_img = np.clip(perturbed_img, 0, 1)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return self.save_image(perturbed_img*225, "temp_adv.jpg")

    def verify_pair(self, img1_path, img2_path):
        """Verify a pair of images using DeepFace"""
        try:
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='opencv',  
                distance_metric='cosine',   
                align=True                  
            )
            return result['verified']
        except Exception as e:
            print(f"Error in verification: {e}")
            return False

    def evaluate_attack(self, attack_type):
        """Evaluate attack performance"""
        results = {
            'true_positive': 0, 'true_negative': 0,
            'false_positive': 0, 'false_negative': 0
        }
        
        for img1_path, img2_path, label in tqdm(self.pairs):
            try:
                # Get target embedding
                target_embedding = self.compute_embedding(img2_path)
                if target_embedding is None:
                    continue
                
                # Apply attack
                if attack_type == "FGSM":
                    adv_img_path = self.apply_fgsm_attack(img1_path, target_embedding, label)
                else:  # PGD
                    adv_img_path = self.apply_pgd_attack(img1_path, target_embedding, label)
                
                # Verify
                prediction = self.verify_pair(adv_img_path, img2_path)
                
                # Update results
                if label == 1:
                    if prediction: 
                        results['true_positive'] += 1
                        print(f"True Positive: {img1_path} - {img2_path}")
                    else: 
                        results['false_negative'] += 1
                        print(f"False Negative: {img1_path} - {img2_path}")
                else:
                    if prediction: 
                        results['false_positive'] += 1
                        print(f"False Positive: {img1_path} - {img2_path}")
                    else: 
                        results['true_negative'] += 1
                        print(f"True Negative: {img1_path} - {img2_path}")
                
                # Clean up
                if os.path.exists(adv_img_path):
                    os.remove(adv_img_path)
                
            except Exception as e:
                print(f"Error processing pair: {e}")
                continue
        
        # Calculate metrics
        total = sum(results.values())
        
        if total == 0:
            return {'accuracy': 0, 'far': 0, 'frr': 0, 'attack_success_rate': 0}        
        return {
            'accuracy': (results['true_positive'] + results['true_negative']) / total,
            'far': results['false_positive'] /  (results['false_positive'] + results['true_negative']),
            'frr': results['false_negative'] / (results['false_negative'] + results['true_positive']),
            'attack_success_rate': (results['false_negative'] + results['false_positive']) / total}
    def run_evaluation(self):
        results = {}
        # Clean performance
        print("Evaluating clean performance...")
        clean_results = {'true_positive': 0, 'true_negative': 0,
                        'false_positive': 0, 'false_negative': 0}
        
        for img1_path, img2_path, label in tqdm(self.pairs):
            prediction = self.verify_pair(img1_path, img2_path)
            if label == 1:
                if prediction: clean_results['true_positive'] += 1
                else: clean_results['false_negative'] += 1
            else:
                if prediction: clean_results['false_positive'] += 1
                else: clean_results['true_negative'] += 1
        
        total = sum(clean_results.values())
        results['clean'] = {
            'accuracy': (clean_results['true_positive'] + clean_results['true_negative']) / total,
            'far': clean_results['false_positive'] / (clean_results['false_positive'] + clean_results['true_negative']),
            'frr': clean_results['false_negative'] / (clean_results['false_negative'] + clean_results['true_positive'])
        }
        # Attack evaluations
        for attack_type in ["FGSM", "PGD"]:
            print(f"\nEvaluating {attack_type} attack...")
            results[attack_type] = self.evaluate_attack(attack_type)
        
        return results

if __name__ == "__main__":
    framework = DeepFaceAttackFramework(
        data_dir='E:/lfw/lfw-py/lfw_funneled',
        model_name="VGG-Face"
    )
    results = framework.run_evaluation()
    
    for scenario, metrics in results.items():
        print(f"\n{scenario} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")