import numpy as np
from deepface import DeepFace
import cv2
from tqdm import tqdm
import os
import tensorflow as tf

class DeepFaceAttackFramework:
    def __init__(self, data_dir, model_name="VGG-Face"):
        self.data_dir = data_dir
        self.model_name = model_name
        self.pairs = self.prepare_pairs()
    
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
        
        # Different person pairs
        for i in range(len(classes)):
            for j in range(i + 1, min(i + 2, len(classes))):
                img1 = os.path.join(self.data_dir, classes[i], 
                                  os.listdir(os.path.join(self.data_dir, classes[i]))[0])
                img2 = os.path.join(self.data_dir, classes[j], 
                                  os.listdir(os.path.join(self.data_dir, classes[j]))[0])
                pairs.append((img1, img2, 0))
        
        return pairs

    def save_image(self, img_array, path="temp.jpg"):
        """Save image array to file"""
        if img_array.dtype != np.uint8:
            img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(path, img_array)
        return path

    def compute_embedding(self, img_path):
        try:
            # DeepFace.represent returns a list of dictionaries
            result = DeepFace.represent(
                img_path=img_path,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='opencv',
                max_faces=1
            )
            
            # Get the first face's embedding (we assume one face per image)
            if not result or len(result) == 0:
                print(f"No faces found in {img_path}")
                return None
                
            # Extract the embedding array from the first face
            embedding = np.array(result[0]['embedding'])
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

    def compute_gradient(self, img_path, target_embedding):
        """Compute numerical gradient"""
        epsilon = 1e-7
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = img.astype(np.float32) / 255.0
        gradient = np.zeros_like(img)
        # Compute gradient for a subset of pixels to speed up computation
        step = 4  # Skip pixels to speed up computation
        temp_path = self.save_image(img)
        pos_embedding = self.compute_embedding(temp_path)
        neg_embedding = self.compute_embedding(temp_path)
        for i in range(0, img.shape[0], step):
            for j in range(0, img.shape[1], step):
                for k in range(img.shape[2]):
                    orig_val = img[i, j, k]
                    # Compute f(x + epsilon)
                    img[i, j, k] = min(1.0, orig_val + epsilon)
                    pos_sim = self.compute_cosine_similarity(target_embedding, pos_embedding)
                    # Compute f(x - epsilon)
                    img[i, j, k] = max(0.0, orig_val - epsilon)
                    neg_sim = self.compute_cosine_similarity(target_embedding, neg_embedding)
                    # Restore original value
                    img[i, j, k] = orig_val
                    
                    # Compute gradient (negative because we want to minimize similarity)
                    gradient[i, j, k] = -(pos_sim - neg_sim) / (2 * epsilon)
                    
                    # Copy gradient to nearby pixels
                    for di in range(step):
                        for dj in range(step):
                            if i + di < img.shape[0] and j + dj < img.shape[1]:
                                gradient[i + di, j + dj, k] = gradient[i, j, k]
        
        if os.path.exists("temp.jpg"):
            os.remove("temp.jpg")
        
        return gradient


    def apply_fgsm_attack(self, img_path, target_embedding, epsilon=0.03):
        """Apply FGSM attack"""
        # Load and normalize image
        img = cv2.imread(img_path)
        img = img.astype(np.float32) / 255.0
        
        # Compute gradient
        gradient = self.compute_gradient(img_path, target_embedding)
        
        # Apply perturbation
        perturbation = epsilon * np.sign(gradient)
        perturbed_img = img + perturbation
        perturbed_img = np.clip(perturbed_img, 0, 1)
        
        # Save perturbed image
        return self.save_image(perturbed_img, "temp_adv.jpg")

    def apply_pgd_attack(self, img_path, target_embedding, epsilon=0.03, 
                        alpha=0.01, steps=10):  # Reduced steps for efficiency
        """Apply PGD attack"""
        # Load and normalize image
        img = cv2.imread(img_path)
        img = img.astype(np.float32) / 255.0
        perturbed_img = img.copy()
        
        for _ in range(steps):
            # Save current perturbed image for gradient computation
            temp_path = self.save_image(perturbed_img)
            
            # Compute gradient
            gradient = self.compute_gradient(temp_path, target_embedding)
            
            # Update image
            perturbation = alpha * np.sign(gradient)
            perturbed_img = perturbed_img + perturbation
            
            # Project back to epsilon-ball and valid image range
            perturbed_img = np.clip(perturbed_img, img - epsilon, img + epsilon)
            perturbed_img = np.clip(perturbed_img, 0, 1)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Save final perturbed image
        return self.save_image(perturbed_img, "temp_adv.jpg")

    def verify_pair(self, img1_path, img2_path):
        """Verify a pair of images using DeepFace"""
        try:
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=self.model_name,
                enforce_detection=False
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
                    adv_img_path = self.apply_fgsm_attack(img1_path, target_embedding)
                else:  # PGD
                    adv_img_path = self.apply_pgd_attack(img1_path, target_embedding)
                
                # Verify
                prediction = self.verify_pair(adv_img_path, img2_path)
                
                # Update results
                if label == 1:
                    if prediction: results['true_positive'] += 1
                    else: results['false_negative'] += 1
                else:
                    if prediction: results['false_positive'] += 1
                    else: results['true_negative'] += 1
                
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
            
        denominator_fp_tn = results['false_positive'] + results['true_negative']
        denominator_fn_tp = results['false_negative'] + results['true_positive']
        
        return {
            'accuracy': (results['true_positive'] + results['true_negative']) / total,
            'far': results['false_positive'] / denominator_fp_tn if denominator_fp_tn > 0 else 0,
            'frr': results['false_negative'] / denominator_fn_tp if denominator_fn_tp > 0 else 0,
            'attack_success_rate': results['false_negative'] / denominator_fn_tp if denominator_fn_tp > 0 else 0
        }
    def run_evaluation(self):
        results = {}
        '''
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
        '''
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