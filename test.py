import torch
import numpy as np
from deepface import DeepFace
from PIL import Image
import torchattacks
from tqdm import tqdm
import os
import cv2

class DeepFaceAdversarialFramework:
    def __init__(self, data_dir, model_name="VGG-Face"):
        """
        Initialize framework with specific DeepFace model
        Available models: "VGG-Face", "Facenet", "OpenFace", "DeepFace", "SFace", "GhostFaceNet"
        """
        self.data_dir = data_dir
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pairs = self.prepare_pairs()
        
    def prepare_pairs(self):
        """Prepare image pairs for verification"""
        pairs = []
        classes = os.listdir(self.data_dir)
        
        # Generate genuine pairs (same person)
        for person in classes:
            person_dir = os.path.join(self.data_dir, person)
            images = os.listdir(person_dir)
            if len(images) >= 2:
                img1 = os.path.join(person_dir, images[0])
                img2 = os.path.join(person_dir, images[1])
                pairs.append((img1, img2, 1))  # 1 indicates same person
                
        # Generate impostor pairs (different people)
        for i in range(len(classes)):
            for j in range(i + 1, min(i + 2, len(classes))):
                img1 = os.path.join(self.data_dir, classes[i], os.listdir(os.path.join(self.data_dir, classes[i]))[0])
                img2 = os.path.join(self.data_dir, classes[j], os.listdir(os.path.join(self.data_dir, classes[j]))[0])
                pairs.append((img1, img2, 0))  # 0 indicates different people
                
        return pairs

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

    def apply_attack(self, image_path, attack_type, epsilon=0.03):
        """Apply adversarial attack to image"""
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        # Apply attack
        if attack_type == "FGSM":
            attack = torchattacks.FGSM(self.model, eps=epsilon)
        elif attack_type == "PGD":
            attack = torchattacks.PGD(self.model, eps=0.3, alpha=2/255, steps=40)
        elif attack_type == "CW":
            attack = torchattacks.CW(self.model, c=1, kappa=0, steps=100)
        
        adv_img = attack(img_tensor, torch.zeros(1))  # Dummy label
        
        # Convert back to image
        adv_img = adv_img.squeeze().permute(1, 2, 0).cpu().numpy()
        adv_img = np.clip(adv_img, 0, 255).astype(np.uint8)
        
        # Save temporary file
        temp_path = "temp_adv.jpg"
        cv2.imwrite(temp_path, adv_img)
        return temp_path

    def evaluate_attack(self, attack_type):
        """Evaluate model under specific attack"""
        results = {
            'true_positive': 0,
            'true_negative': 0,
            'false_positive': 0,
            'false_negative': 0
        }
        
        for img1_path, img2_path, label in tqdm(self.pairs):
            # Apply attack to first image
            adv_img_path = self.apply_attack(img1_path, attack_type)
            
            # Verify pairs
            prediction = self.verify_pair(adv_img_path, img2_path)
            
            # Update results
            if label == 1:  # Same person
                if prediction:
                    results['true_positive'] += 1
                else:
                    results['false_negative'] += 1
            else:  # Different person
                if prediction:
                    results['false_positive'] += 1
                else:
                    results['true_negative'] += 1
                    
            # Clean up temporary file
            os.remove(adv_img_path)
            
        # Calculate metrics
        total = sum(results.values())
        metrics = {
            'accuracy': (results['true_positive'] + results['true_negative']) / total,
            'far': results['false_positive'] / (results['false_positive'] + results['true_negative']),
            'frr': results['false_negative'] / (results['false_negative'] + results['true_positive']),
            'attack_success_rate': results['false_negative'] / (results['true_positive'] + results['false_negative'])
        }
        
        return metrics

    def run_full_evaluation(self):
        """Run complete evaluation suite"""
        attacks = ["FGSM", "PGD", "CW"]
        results = {}
        
        # Evaluate clean performance first
        print("Evaluating clean performance...")
        clean_results = {
            'true_positive': 0,
            'true_negative': 0,
            'false_positive': 0,
            'false_negative': 0
        }
        
        for img1_path, img2_path, label in tqdm(self.pairs):
            prediction = self.verify_pair(img1_path, img2_path)
            if label == 1:
                if prediction:
                    clean_results['true_positive'] += 1
                else:
                    clean_results['false_negative'] += 1
            else:
                if prediction:
                    clean_results['false_positive'] += 1
                else:
                    clean_results['true_negative'] += 1
        
        total = sum(clean_results.values())
        results['clean'] = {
            'accuracy': (clean_results['true_positive'] + clean_results['true_negative']) / total,
            'far': clean_results['false_positive'] / (clean_results['false_positive'] + clean_results['true_negative']),
            'frr': clean_results['false_negative'] / (clean_results['false_negative'] + clean_results['true_positive'])
        }
        
        # Evaluate each attack
        for attack_type in attacks:
            print(f"\nEvaluating {attack_type} attack...")
            results[attack_type] = self.evaluate_attack(attack_type)
        
        return results

# Usage Example
if __name__ == "__main__":
    framework = DeepFaceAdversarialFramework(
        data_dir='E:/lfw/lfw-py/lfw_funneled',
        model_name="VGG-Face"  # Can be changed to other models
    )
    
    results = framework.run_full_evaluation()
    
    # Print results
    print("\nEvaluation Results:")
    for scenario, metrics in results.items():
        print(f"\n{scenario} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")