import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchattacks
import numpy as np
import cv2
from tqdm import tqdm
import os

class VGG_16(nn.Module):
    """VGG-Face implementation"""
    def __init__(self):
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)
        return self.fc8(x)

class LFWDataset(Dataset):
    """LFW dataset with proper VGG-Face preprocessing"""
    def __init__(self, image_pairs):
        self.image_pairs = image_pairs
        # VGG mean values for preprocessing
        self.mean = torch.Tensor([129.1863, 104.7624, 93.5940]).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.image_pairs)

    def preprocess_image(self, img_path):
        """Preprocess image according to VGG-Face requirements"""
        img = cv2.imread(img_path)
        # Resize to 224x224 if needed
        if img.shape[:2] != (224, 224):
            img = cv2.resize(img, (224, 224))
        # Convert to torch tensor and adjust dimensions
        img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)
        # Subtract mean
        img = img - self.mean
        return img.squeeze(0)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.image_pairs[idx]
        img1 = self.preprocess_image(img1_path)
        img2 = self.preprocess_image(img2_path)
        return img1, img2, torch.tensor(label, dtype=torch.long)

class VGGAttackFramework:
    def __init__(self, data_dir, model_path, device='cuda'):
        self.data_dir = data_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = VGG_16().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Prepare dataset
        self.pairs = self.prepare_pairs()
        self.dataset = LFWDataset(self.pairs)
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=False)
        
        # Initialize attacks
        self.fgsm = torchattacks.FGSM(self.model, eps=0.3)
        self.pgd = torchattacks.PGD(self.model, eps=0.3, alpha=0.01, steps=40)

    def prepare_pairs(self):
        """Prepare pairs from LFW dataset"""
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

    def verify_pair(self, img1, img2):
        """Verify a pair of images using model predictions"""
        with torch.no_grad():
            # Get predictions
            pred1 = F.softmax(self.model(img1), dim=1)
            pred2 = F.softmax(self.model(img2), dim=1)
            
            # Get predicted classes
            class1 = torch.argmax(pred1, dim=1)
            class2 = torch.argmax(pred2, dim=1)
            
            return class1 == class2

    def evaluate_attack(self, attack_type="fgsm"):
        """Evaluate attack performance"""
        results = {
            'true_positive': 0, 'true_negative': 0,
            'false_positive': 0, 'false_negative': 0
        }
        
        attack = self.fgsm if attack_type.lower() == "fgsm" else self.pgd
        
        for img1, img2, label in tqdm(self.dataloader):
            img1, img2 = img1.to(self.device), img2.to(self.device)
            
            # Generate adversarial examples
            adv_img1 = attack(img1, img2)
            
            # Verify pairs
            prediction = self.verify_pair(adv_img1, img2)
            
            # Update results
            batch_size = label.size(0)
            for i in range(batch_size):
                if label[i] == 1:
                    if prediction[i]:
                        results['true_positive'] += 1
                    else:
                        results['false_negative'] += 1
                else:
                    if prediction[i]:
                        results['false_positive'] += 1
                    else:
                        results['true_negative'] += 1
        
        # Calculate metrics
        total = sum(results.values())
        return {
            'accuracy': (results['true_positive'] + results['true_negative']) / total,
            'far': results['false_positive'] / (results['false_positive'] + results['true_negative']),
            'frr': results['false_negative'] / (results['false_negative'] + results['true_positive']),
            'attack_success_rate': (results['false_negative'] + results['false_positive']) / total
        }

    def run_evaluation(self):
        """Run full evaluation with clean and attacked performance"""
        results = {}
        
        # Clean performance
        print("Evaluating clean performance...")
        clean_results = {'true_positive': 0, 'true_negative': 0,
                        'false_positive': 0, 'false_negative': 0}
        
        with torch.no_grad():
            for img1, img2, label in tqdm(self.dataloader):
                img1, img2 = img1.to(self.device), img2.to(self.device)
                prediction = self.verify_pair(img1, img2)
                
                batch_size = label.size(0)
                for i in range(batch_size):
                    if label[i] == 1:
                        if prediction[i]:
                            clean_results['true_positive'] += 1
                        else:
                            clean_results['false_negative'] += 1
                    else:
                        if prediction[i]:
                            clean_results['false_positive'] += 1
                        else:
                            clean_results['true_negative'] += 1
        
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
    framework = VGGAttackFramework(
        data_dir='E:/lfw/lfw-py/lfw_funneled',
        model_path='E:/AdversarialAttack-2/VGGFace/vgg_face_dag.pth'
    )
    results = framework.run_evaluation()
    
    for scenario, metrics in results.items():
        print(f"\n{scenario} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")