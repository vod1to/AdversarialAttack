import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
import os,sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from Model.Architecture.VGGFaceArchitecture import VGGFace


class VGGAttackFramework:
    def __init__(self, data_dir, model_path, device='cuda'):
        self.data_dir = data_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.pairs = self.prepare_pairs()
        
        # Initialize model
        self.model = VGGFace().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()        
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
            if len(pairs) < 0:
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

    def verify_pair(self, img1_path, img2_path, threshold=1.2):
        # Read and preprocess images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        img1 = cv2.resize(img1, (224, 224))
        img2 = cv2.resize(img2, (224, 224))
        
        # Convert to torch tensor and normalize
        img1 = torch.Tensor(img1).float().permute(2, 0, 1).view(1, 3, 224, 224)
        img2 = torch.Tensor(img2).float().permute(2, 0, 1).view(1, 3, 224, 224)
        
        # VGG Face mean subtraction
        mean = torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).double().view(1, 3, 1, 1)

        img1 -= mean
        img2 -= mean
        
        # Move to device
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        # Get features from fc7 layer
        self.model.eval()
        with torch.no_grad():
            # Forward pass until fc7 for both images
            feat1 = self.model.get_features(img1)
            feat2 = self.model.get_features(img2)
            feat1 = F.normalize(feat1, p=2, dim=1)
            feat2 = F.normalize(feat2, p=2, dim=1)
            # Compute L2 distance
            l2_distance = torch.norm(feat1 - feat2, p=2).item()
            return l2_distance < threshold
    def generateFGSMAttack(self, img1_path, img2_path, label = None):

        return True
    def generatePGDAttack(self, img1_path, img2_path, label = None):
        
        return True
    def evaluate_attack(self, attack_type):
        """Evaluate attack performance"""
        results = {
            'true_positive': 0, 'true_negative': 0,
            'false_positive': 0, 'false_negative': 0
        }
        
        for img1_path, img2_path, label in tqdm(self.pairs):
            try:                
                # Apply attack
                if attack_type == "FGSM":
                    adv_img_path = self.generateFGSMAttack(img1_path, img2_path, label)
                else:  # PGD
                    adv_img_path = self.generatePGDAttack(img1_path, img2_path, label)
                
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
        return results
        # Attack evaluations
        """
        for attack_type in ["FGSM", "PGD"]:
            print(f"\nEvaluating {attack_type} attack...")
            results[attack_type] = self.evaluate_attack(attack_type)
        
        return results"""

if __name__ == "__main__":
    framework = VGGAttackFramework(
        data_dir='E:/lfw/lfw-py/lfw_funneled',
        model_path='E:/AdversarialAttack-2/Model/Weights/vgg_face_dag.pth'
    )
    results = framework.run_evaluation()
    
    for scenario, metrics in results.items():
        print(f"\n{scenario} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")