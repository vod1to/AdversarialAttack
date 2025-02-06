import numpy as np
from deepface import DeepFace
import cv2
from tqdm import tqdm
import os

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
                img1 = os.path.join(self.data_dir, classes[i], os.listdir(os.path.join(self.data_dir, classes[i]))[0])
                img2 = os.path.join(self.data_dir, classes[j], os.listdir(os.path.join(self.data_dir, classes[j]))[0])
                pairs.append((img1, img2, 0))
        
        return pairs

    def apply_fgsm_attack(self, img_path, epsilon=0.03):
        # Load image
        img = cv2.imread(img_path)
        
        # Get gradient direction using simple edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.stack([gradient_x, gradient_y, gradient_y], axis=-1)
        gradient = np.sign(gradient)
        
        # Apply perturbation
        perturbed = img + epsilon * gradient
        perturbed = np.clip(perturbed, 0, 255).astype(np.uint8)
        
        # Save temporary file
        temp_path = "temp_adv.jpg"
        cv2.imwrite(temp_path, perturbed)
        return temp_path

    def apply_pgd_attack(self, img_path, epsilon=0.3, alpha=2/255, steps=40):
        img = cv2.imread(img_path)
        perturbed = img.copy()
        
        for _ in range(steps):
            # Get gradient
            gray = cv2.cvtColor(perturbed, cv2.COLOR_BGR2GRAY)
            gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient = np.stack([gradient_x, gradient_y, gradient_y], axis=-1)
            gradient = np.sign(gradient)
            
            # Update with projected gradient descent
            perturbed = perturbed + alpha * gradient
            perturbed = np.clip(perturbed - img, -epsilon, epsilon) + img
            perturbed = np.clip(perturbed, 0, 255).astype(np.uint8)
        
        temp_path = "temp_adv.jpg"
        cv2.imwrite(temp_path, perturbed)
        return temp_path

    def verify_pair(self, img1_path, img2_path):
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
        results = {
            'true_positive': 0, 'true_negative': 0,
            'false_positive': 0, 'false_negative': 0
        }
        
        for img1_path, img2_path, label in tqdm(self.pairs):
            # Apply attack based on type
            if attack_type == "FGSM":
                adv_img_path = self.apply_fgsm_attack(img1_path)
            elif attack_type == "PGD":
                adv_img_path = self.apply_pgd_attack(img1_path)
            
            prediction = self.verify_pair(adv_img_path, img2_path)
            
            if label == 1:
                if prediction: results['true_positive'] += 1
                else: results['false_negative'] += 1
            else:
                if prediction: results['false_positive'] += 1
                else: results['true_negative'] += 1
            
            os.remove(adv_img_path)
        
        total = sum(results.values())
        return {
            'accuracy': (results['true_positive'] + results['true_negative']) / total,
            'far': results['false_positive'] / (results['false_positive'] + results['true_negative']),
            'frr': results['false_negative'] / (results['false_negative'] + results['true_positive']),
            'attack_success_rate': results['false_negative'] / (results['true_positive'] + results['false_negative'])
        }

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